#[derive(Debug, Clone)]
pub struct ErrorStruct {
    pub source: &'static dyn std::error::Error
}

impl std::fmt::Display for ErrorStruct {

}

