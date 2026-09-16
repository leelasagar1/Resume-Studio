"""Run the single-user local application. Run setup.sh / setup.ps1 first.

Binds to 127.0.0.1 by default. HOST=0.0.0.0 is only for the Docker image,
where docker-compose publishes the port on the host's loopback address.
"""
import os
import uvicorn

if __name__ == '__main__':
    uvicorn.run('app.main:app', host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8765')))
