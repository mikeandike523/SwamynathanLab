use serde_json;


use crate::error_type::ErrorType;

pub mod IndexError {

    use crate::error_type::ErrorType;

    pub fn new(message: Option<String>) -> ErrorType {
        ErrorType::new(String::from("index_error"), message)
    }

}

pub mod JSONDecodeError {

    pub mod GenericSerdeJSONError {
        use crate::error_type::ErrorType;

        pub fn new(serde_json_error:serde_json::Error) -> ErrorType {
            ErrorType::new(String::from("json_decode_error.generic_serde_json_error"),
                           Some(format!("{}",serde_json_error))
            )
        }
    }

    pub mod IncorrectTypeError {
        use crate::error_type::ErrorType;

        pub fn new(field_name:String, field_required_type:String) -> ErrorType {
            ErrorType::new(String::from("json_decode_error.incorrect_type_error"),
                Some(format!("Field \"{}\" had required type {} but an incompatible serde type was found.",field_name,field_required_type))
            )
        }
    }


}