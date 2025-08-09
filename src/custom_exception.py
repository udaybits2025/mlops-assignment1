import traceback
import sys

class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    # We are adding static method because we dont need to create custom Exception multiple times
    @staticmethod
    def get_detailed_error_message(error_message, error_detail:sys):

        _, _, exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename # This will fetch the filename of the error
        line_number = exc_tb.tb_lineno # This will extract the line number of the error

        return f"Error in {file_name} , at line {line_number} : {error_message}"
    
    def __str__(self):
        return self.error_message