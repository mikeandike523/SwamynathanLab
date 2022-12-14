#[derive(Debug)]
pub struct ErrorType {
    code: String,
    message: Option<String>
}

impl ErrorType {
    pub fn new(code:String, message:Option<String>) -> ErrorType{
        ErrorType {
            code,
            message
        }
    }
}