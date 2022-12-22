use std::fmt;
use std::error::Error;


#[derive(Debug)]
pub struct Io3Error {
    source: Option<Box<(dyn Error)>>
} 

impl fmt::Display for Io3Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,"{}", format!("{:?}", self))
    }
}

impl Error for Io3Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.source {
            Some(s) => {
                Some(&**s)
            },
            None=>None
        }
    }
}

impl Io3Error {
    pub fn new(source: Option<Box<dyn Error>>) -> Io3Error {
        Io3Error {
            source: source
        }
    }
}

impl std::convert::Into<pyo3::PyErr> for Io3Error {
    fn into(self) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(format!("{}", self))
    }
}