import os
os.environ["WORKSHOP_MODE"] = "student"
from src.app.viewer import main
if __name__ == "__main__":
    main()
