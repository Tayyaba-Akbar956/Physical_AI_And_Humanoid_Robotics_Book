from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from typing import Callable, Awaitable
import traceback
import logging
from datetime import datetime, UTC
import json


class ErrorHandlingMiddleware:
    """
    Custom middleware for comprehensive error handling and logging
    """
    
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            # Process the request - JUST AWAIT, DON'T RETURN
            await self.app(scope, receive, send)
            
        except HTTPException as e:
            # Handle HTTP exceptions
            await self.handle_http_exception(scope, receive, send, e)
            
        except Exception as e:
            # Handle general exceptions
            request = Request(scope)
            await self.handle_general_exception(scope, receive, send, e, request)


def add_security_headers(app):
    """
    Add security headers to all responses
    """
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


def add_rate_limiting(app):
    """
    Add basic rate limiting functionality
    """
    # This is a simplified implementation 
    # In a production environment, you would use a more robust solution like slowapi
    request_counts = {}
    
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # For this example, we'll implement a simple rate limiter
        # In a real application, you'd want to use a distributed solution like Redis
        client_ip = request.client.host
        current_time = datetime.now(UTC).timestamp()

        # Initialize client record if needed
        if client_ip not in request_counts:
            request_counts[client_ip] = []

        # Clean old requests (older than 1 minute)
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip]
            if current_time - req_time < 60
        ]

        # Add current request
        request_counts[client_ip].append(current_time)

        # Check if rate limit exceeded (100 requests per minute)
        if len(request_counts[client_ip]) > 100:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "type": "RateLimitExceeded",
                        "message": "Rate limit exceeded. Please try again later.",
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                }
            )
        
        response = await call_next(request)
        return response