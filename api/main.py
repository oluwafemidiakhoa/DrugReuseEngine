from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

# Import our routers
from api.routers import drugs, diseases, knowledge_graph, auth
from api.utils.rate_limiter import rate_limit_middleware
from api.security.auth import get_current_active_user

# Create the FastAPI application
app = FastAPI(
    title="Drug Repurposing Engine API",
    description="A comprehensive computational platform for identifying and evaluating potential new uses for existing drugs",
    version="1.0.0",
    # Turn off the automatic docs for custom handling
    docs_url=None,
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
@app.middleware("http")
async def rate_limiting(request: Request, call_next):
    """Apply rate limiting to all requests"""
    return await rate_limit_middleware(request, call_next)

# Include the routers
app.include_router(auth.router)
app.include_router(drugs.router)
app.include_router(diseases.router)
app.include_router(knowledge_graph.router)

# Custom exception handler for rate limiting
@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc):
    """Handle rate limiting exceptions"""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
        headers={"Retry-After": "60"}
    )

# Custom OpenAPI documentation endpoints
@app.get("/docs", include_in_schema=False)
async def get_documentation():
    """Serve Swagger UI documentation"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Drug Repurposing Engine API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def get_redoc_documentation():
    """Serve ReDoc documentation"""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title="Drug Repurposing Engine API - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Serve OpenAPI JSON schema"""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    """API root - provides basic information"""
    return {
        "name": "Drug Repurposing Engine API",
        "version": app.version,
        "description": "A comprehensive computational platform for identifying and evaluating potential new uses for existing drugs",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# Health check endpoint
@app.get("/health", tags=["status"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "version": app.version}

# Protected test endpoint
@app.get("/protected", tags=["status"])
async def protected_route(current_user = Depends(get_current_active_user)):
    """Test endpoint that requires authentication"""
    return {
        "message": f"Hello, {current_user.username}! You have access to protected routes.",
        "user": current_user
    }