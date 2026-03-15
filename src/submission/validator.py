from typing import Dict, Any

class SubmissionValidationError(Exception):
    pass

def validate_submission(submission: Dict[str, Any]) -> None:
    if not isinstance(submission, dict):
        raise SubmissionValidationError("Submission must be a dictionary")
    
    required_fields = ["team_email", "submission_name", "answers"]
    for field in required_fields:
        if field not in submission:
            raise SubmissionValidationError(f"Missing required field: {field}")
    
    if not isinstance(submission["answers"], list) or len(submission["answers"]) == 0:
        raise SubmissionValidationError("'answers' must be a non-empty list")
    
    for idx, ans in enumerate(submission["answers"]):
        if not isinstance(ans, dict):
            raise SubmissionValidationError(f"Answer #{idx} must be a dictionary")
        
        if "value" not in ans:
            raise SubmissionValidationError(f"Answer #{idx} missing 'value'")
        if "references" not in ans:
            raise SubmissionValidationError(f"Answer #{idx} missing 'references'")
        
        if not isinstance(ans["references"], list):
            raise SubmissionValidationError(f"Answer #{idx}: 'references' must be a list")
        
        for ref_idx, ref in enumerate(ans["references"]):
            if not isinstance(ref, dict):
                raise SubmissionValidationError(f"Answer #{idx}, ref #{ref_idx}: must be a dictionary")
            if "pdf_sha1" not in ref or "page_index" not in ref:
                raise SubmissionValidationError(f"Answer #{idx}, ref #{ref_idx}: missing pdf_sha1 or page_index")
            if not isinstance(ref["pdf_sha1"], str):
                raise SubmissionValidationError(f"Answer #{idx}, ref #{ref_idx}: pdf_sha1 must be string")
            if not isinstance(ref["page_index"], int):
                raise SubmissionValidationError(f"Answer #{idx}, ref #{ref_idx}: page_index must be integer")
    
    for idx, ans in enumerate(submission["answers"]):
        val = ans["value"]
        if isinstance(val, str) and val.lower() in ['yes', 'no', 'true', 'false']:
            if val.lower() not in ['yes', 'no']:
                raise SubmissionValidationError(f"Answer #{idx}: boolean value must be 'yes' or 'no', got '{val}'")