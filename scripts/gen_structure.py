import os
from pathlib import Path

# Directories
dirs = [
    'backend/app/models',
    'backend/app/routes',
    'backend/app/services',
    'backend/app/utils',
    'backend/scripts',
    'frontend/src/components',
    'frontend/src/pages',
    'frontend/src/services',
    'frontend/public',
]

# Files
files = [
    'backend/app/main.py',
    'backend/app/models/schema.py',
    'backend/app/routes/reviews.py',
    'backend/app/routes/users.py',
    'backend/app/services/recommend.py',
    'backend/scripts/train_model.py',
    'frontend/src/App.js',
    'frontend/src/index.js',
    'frontend/src/pages/Home.js',
    'frontend/src/pages/Products.js',
    'frontend/src/pages/Reviews.js',
    'frontend/src/pages/Recommendations.js',
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).parent.mkdir(parents=True, exist_ok=True)
    Path(f).touch()
print("✅ Project structure created")
