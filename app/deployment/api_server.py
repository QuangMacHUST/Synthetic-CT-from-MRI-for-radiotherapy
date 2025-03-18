#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ... existing code ...
def start_server(
    model: str,
    region: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    api_key_param: Optional[str] = None,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
    config: Optional[ConfigManager] = None
) -> None:
    """
    Start API server.
    
    Args:
        model: Conversion model to use
        region: Anatomical region
        host: Host to run server on
        port: Port to run server on
        workers: Number of worker processes
        api_key_param: API key for authentication
        ssl_cert: Path to SSL certificate
        ssl_key: Path to SSL key
        config: Configuration manager
    """
    global api_key
    api_key = api_key_param
    
    # Initialize models
    initialize_models(model, region, config)
    
    # Start server
    uvicorn.run(
        "app.deployment.api_server:app",
        host=host,
        port=port,
        workers=workers,
        ssl_certfile=ssl_cert,
        ssl_keyfile=ssl_key
    )
# ... existing code ...