//! STAR/CIF Document Object Model.
//!
//! Provides an untyped representation of any CIF/STAR file, with a query API
//! for extracting data by tag name.

/// A parsed CIF/STAR document containing one or more data blocks.
#[derive(Debug, Clone)]
pub struct Document {
    pub blocks: Vec<Block>,
}

/// A data block (`data_NAME`) containing key-value pairs, loops, and save frames.
#[derive(Debug, Clone)]
pub struct Block {
    pub name: String,
    pub pairs: Vec<(String, Value)>,
    pub loops: Vec<Loop>,
    pub frames: Vec<Block>, // save frames (rare, only in dictionaries)
}

/// A looped data table: named columns with row-major values.
#[derive(Debug, Clone)]
pub struct Loop {
    /// Column names (tags).
    pub tags: Vec<String>,
    /// Row-major flat array of values. Length = `tags.len() * nrows()`.
    pub values: Vec<Value>,
}

/// A CIF data value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A string value (unquoted, single-quoted, double-quoted, or semicolon text).
    Str(String),
    /// The inapplicable marker `.`.
    Inapplicable,
    /// The unknown marker `?`.
    Unknown,
}

impl Value {
    /// Returns the string content, or `None` for `.` / `?`.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Tries to parse the value as `f64`.
    ///
    /// Handles CIF uncertainty notation like `50.123(4)` by stripping the
    /// parenthesized uncertainty before parsing.
    pub fn as_f64(&self) -> Option<f64> {
        let s = self.as_str()?;
        let s = match s.find('(') {
            Some(idx) => &s[..idx],
            None => s,
        };
        s.parse().ok()
    }

    /// Tries to parse the value as `i32`.
    pub fn as_i32(&self) -> Option<i32> {
        self.as_str()?.parse().ok()
    }

    /// Returns `true` if this is a `Str` value (not `.` or `?`).
    pub fn is_present(&self) -> bool {
        matches!(self, Value::Str(_))
    }
}

impl Loop {
    /// Number of rows in this loop.
    pub fn nrows(&self) -> usize {
        if self.tags.is_empty() {
            0
        } else {
            self.values.len() / self.tags.len()
        }
    }

    /// Find the column index for a tag (case-insensitive).
    pub fn column_index(&self, tag: &str) -> Option<usize> {
        self.tags
            .iter()
            .position(|t| t.eq_ignore_ascii_case(tag))
    }
}

impl Block {
    /// Get a single key-value pair by tag name (case-insensitive).
    pub fn get(&self, tag: &str) -> Option<&Value> {
        self.pairs
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(tag))
            .map(|(_, v)| v)
    }

    /// Find the loop containing a given tag (case-insensitive).
    pub fn find_loop(&self, tag: &str) -> Option<&Loop> {
        self.loops
            .iter()
            .find(|lp| lp.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
    }

    /// Get a single column from whatever loop contains it.
    pub fn column(&self, tag: &str) -> Option<ColumnIter<'_>> {
        let lp = self.find_loop(tag)?;
        let col_idx = lp.column_index(tag)?;
        Some(ColumnIter {
            lp,
            col_idx,
            row: 0,
        })
    }

    /// Get multiple columns from the same loop, for row-wise iteration.
    ///
    /// Returns `None` if any tag is missing or if the tags span different loops.
    pub fn columns(&self, tags: &[&str]) -> Option<Columns<'_>> {
        if tags.is_empty() {
            return None;
        }
        let lp = self.find_loop(tags[0])?;
        let mut col_indices = Vec::with_capacity(tags.len());
        for tag in tags {
            col_indices.push(lp.column_index(tag)?);
        }
        Some(Columns { lp, col_indices })
    }
}

/// Iterator over a single column's values.
pub struct ColumnIter<'a> {
    lp: &'a Loop,
    col_idx: usize,
    row: usize,
}

impl<'a> Iterator for ColumnIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.lp.nrows() {
            return None;
        }
        let stride = self.lp.tags.len();
        let idx = self.row * stride + self.col_idx;
        self.row += 1;
        Some(&self.lp.values[idx])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.lp.nrows() - self.row;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ColumnIter<'_> {}

/// Multi-column accessor for row-wise iteration over a loop.
pub struct Columns<'a> {
    lp: &'a Loop,
    col_indices: Vec<usize>,
}

impl<'a> Columns<'a> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.lp.nrows()
    }

    /// Number of selected columns.
    pub fn ncols(&self) -> usize {
        self.col_indices.len()
    }

    /// Get the value at `(row, col)` where `col` indexes into the requested tags.
    pub fn get(&self, row: usize, col: usize) -> &'a Value {
        let stride = self.lp.tags.len();
        &self.lp.values[row * stride + self.col_indices[col]]
    }

    /// Iterate over rows, yielding a `Vec<&Value>` per row.
    pub fn iter(&self) -> RowIter<'a> {
        RowIter {
            lp: self.lp,
            col_indices: self.col_indices.clone(),
            row: 0,
        }
    }
}

/// Row-wise iterator over selected columns of a loop.
pub struct RowIter<'a> {
    lp: &'a Loop,
    col_indices: Vec<usize>,
    row: usize,
}

impl<'a> Iterator for RowIter<'a> {
    type Item = Vec<&'a Value>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.lp.nrows() {
            return None;
        }
        let stride = self.lp.tags.len();
        let base = self.row * stride;
        let row: Vec<&Value> = self
            .col_indices
            .iter()
            .map(|&ci| &self.lp.values[base + ci])
            .collect();
        self.row += 1;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.lp.nrows() - self.row;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RowIter<'_> {}
