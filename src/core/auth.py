from fastapi import Header, HTTPException


def make_verify_api_key(settings):
    def verify_api_key(x_api_key: str = Header(default="")):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return verify_api_key
