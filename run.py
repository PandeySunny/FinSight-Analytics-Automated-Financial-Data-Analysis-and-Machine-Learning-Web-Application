#!/usr/bin/env python
"""
Flask app runner for production/deployment
"""
import os
from app import app

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.getenv("PORT", 5000))
    
    # For production, bind to 0.0.0.0 to accept external connections
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,  # Always False in production
        threaded=True
    )
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

