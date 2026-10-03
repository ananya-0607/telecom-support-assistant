from telecom_support.llm import ServiceError

def validate_citations(resolution, evidence):
    allowed = {item['citation_id'] for item in evidence}
    if resolution.status == 'suggested_resolution' and not resolution.steps:
        raise ServiceError('Resolution had no supported steps.')
    if resolution.status == 'insufficient_evidence' and resolution.steps:
        raise ServiceError('Insufficient-evidence response must not propose resolution steps.')
    for step in resolution.steps:
        if not step.instruction.strip() or any(c not in allowed for c in step.citations):
            raise ServiceError('Resolution contained an invalid source citation or empty step.')
    return resolution
