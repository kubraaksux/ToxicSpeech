"""
Optimization Utilities
Additional tools for bundle size, image optimization, and deployment improvements
"""

import os
import gzip
import shutil
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import pickle
import joblib


class BundleSizeOptimizer:
    """Tools for reducing bundle size and optimizing assets."""
    
    @staticmethod
    def optimize_images(image_dir: str, output_dir: str = None, quality: int = 85) -> Dict:
        """Optimize PNG/JPG images for web deployment."""
        if output_dir is None:
            output_dir = image_dir + "_optimized"
        
        Path(output_dir).mkdir(exist_ok=True)
        results = {'original_size': 0, 'optimized_size': 0, 'files_processed': 0}
        
        for image_file in Path(image_dir).glob("*.png"):
            # Load and optimize image
            with Image.open(image_file) as img:
                # Convert to RGB if PNG with transparency for JPG conversion
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                # Get original size
                original_size = image_file.stat().st_size
                
                # Save optimized version
                output_path = Path(output_dir) / f"{image_file.stem}_optimized.jpg"
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                # Get new size
                optimized_size = output_path.stat().st_size
                
                results['original_size'] += original_size
                results['optimized_size'] += optimized_size
                results['files_processed'] += 1
                
                print(f"Optimized {image_file.name}: {original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB "
                      f"({(1-optimized_size/original_size)*100:.1f}% reduction)")
        
        total_reduction = (1 - results['optimized_size'] / results['original_size']) * 100
        print(f"\nTotal optimization: {results['original_size']/1024:.1f}KB → {results['optimized_size']/1024:.1f}KB "
              f"({total_reduction:.1f}% reduction)")
        
        return results
    
    @staticmethod
    def compress_embeddings(embeddings: np.ndarray, output_path: str, compression_level: int = 6) -> Dict:
        """Compress and save embeddings for faster loading."""
        # Save with different compression methods
        results = {}
        
        # Original numpy save
        np.save(output_path + '_numpy.npy', embeddings)
        numpy_size = Path(output_path + '_numpy.npy').stat().st_size
        
        # Compressed numpy save
        np.savez_compressed(output_path + '_compressed.npz', embeddings=embeddings)
        npz_size = Path(output_path + '_compressed.npz').stat().st_size
        
        # Joblib with compression
        joblib.dump(embeddings, output_path + '_joblib.pkl', compress=compression_level)
        joblib_size = Path(output_path + '_joblib.pkl').stat().st_size
        
        # Gzip compressed pickle
        with gzip.open(output_path + '_gzip.pkl.gz', 'wb', compresslevel=compression_level) as f:
            pickle.dump(embeddings, f)
        gzip_size = Path(output_path + '_gzip.pkl.gz').stat().st_size
        
        results = {
            'numpy_size': numpy_size,
            'npz_size': npz_size,
            'joblib_size': joblib_size,
            'gzip_size': gzip_size,
            'best_compression': min(npz_size, joblib_size, gzip_size),
            'compression_ratio': numpy_size / min(npz_size, joblib_size, gzip_size)
        }
        
        print(f"Embedding compression results:")
        print(f"  Original (numpy): {numpy_size/1024/1024:.1f} MB")
        print(f"  NPZ compressed: {npz_size/1024/1024:.1f} MB ({(1-npz_size/numpy_size)*100:.1f}% reduction)")
        print(f"  Joblib compressed: {joblib_size/1024/1024:.1f} MB ({(1-joblib_size/numpy_size)*100:.1f}% reduction)")
        print(f"  Gzip compressed: {gzip_size/1024/1024:.1f} MB ({(1-gzip_size/numpy_size)*100:.1f}% reduction)")
        
        return results
    
    @staticmethod
    def clean_jupyter_notebook(notebook_path: str, output_path: str = None) -> Dict:
        """Clean Jupyter notebook of outputs and unnecessary metadata."""
        import json
        
        if output_path is None:
            output_path = notebook_path.replace('.ipynb', '_cleaned.ipynb')
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        original_size = len(json.dumps(notebook))
        cells_cleaned = 0
        
        # Clean cell outputs and metadata
        for cell in notebook.get('cells', []):
            if 'outputs' in cell:
                cell['outputs'] = []
                cells_cleaned += 1
            if 'execution_count' in cell:
                cell['execution_count'] = None
            if 'metadata' in cell:
                cell['metadata'] = {}
        
        # Clean notebook metadata
        if 'metadata' in notebook:
            # Keep only essential metadata
            essential_keys = ['kernelspec', 'language_info']
            notebook['metadata'] = {k: v for k, v in notebook['metadata'].items() if k in essential_keys}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, separators=(',', ': '))
        
        cleaned_size = len(json.dumps(notebook))
        reduction = (1 - cleaned_size / original_size) * 100
        
        results = {
            'original_size': original_size,
            'cleaned_size': cleaned_size,
            'size_reduction_percent': reduction,
            'cells_cleaned': cells_cleaned
        }
        
        print(f"Notebook cleaned: {original_size/1024:.1f}KB → {cleaned_size/1024:.1f}KB "
              f"({reduction:.1f}% reduction, {cells_cleaned} cells cleaned)")
        
        return results


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def optimize_numpy_arrays(arrays: List[np.ndarray], dtype: str = 'float32') -> List[np.ndarray]:
        """Convert arrays to more memory-efficient dtypes."""
        optimized_arrays = []
        total_original_size = 0
        total_optimized_size = 0
        
        for arr in arrays:
            original_size = arr.nbytes
            total_original_size += original_size
            
            # Convert to specified dtype
            optimized_arr = arr.astype(dtype)
            optimized_size = optimized_arr.nbytes
            total_optimized_size += optimized_size
            
            optimized_arrays.append(optimized_arr)
            
            print(f"Array optimized: {original_size/1024/1024:.1f}MB → {optimized_size/1024/1024:.1f}MB "
                  f"({(1-optimized_size/original_size)*100:.1f}% reduction)")
        
        total_reduction = (1 - total_optimized_size / total_original_size) * 100
        print(f"Total memory reduction: {total_reduction:.1f}%")
        
        return optimized_arrays
    
    @staticmethod
    def create_memory_efficient_embeddings(
        texts: List[str], 
        embedding_generator, 
        chunk_size: int = 100,
        save_path: str = None
    ) -> str:
        """Generate embeddings in chunks to reduce memory usage."""
        if save_path is None:
            save_path = "embeddings_chunked.npz"
        
        all_embeddings = []
        
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            chunk_embeddings = embedding_generator.generate_embeddings_batch(chunk_texts)
            
            # Convert to float32 for memory efficiency
            chunk_embeddings = chunk_embeddings.astype('float32')
            all_embeddings.append(chunk_embeddings)
            
            print(f"Processed chunk {i//chunk_size + 1}/{(len(texts)-1)//chunk_size + 1}")
            
            # Clear memory
            del chunk_embeddings
        
        # Combine and save
        final_embeddings = np.vstack(all_embeddings)
        np.savez_compressed(save_path, embeddings=final_embeddings)
        
        print(f"Embeddings saved to {save_path} ({final_embeddings.nbytes/1024/1024:.1f} MB)")
        
        return save_path


class DeploymentOptimizer:
    """Optimization tools for deployment."""
    
    @staticmethod
    def create_requirements_minimal(
        full_requirements_path: str, 
        used_packages: List[str],
        output_path: str = "requirements_minimal.txt"
    ) -> str:
        """Create minimal requirements file with only used packages."""
        with open(full_requirements_path, 'r') as f:
            all_requirements = f.readlines()
        
        minimal_requirements = []
        
        for req in all_requirements:
            package_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
            if package_name in used_packages or package_name.startswith('#'):
                minimal_requirements.append(req)
        
        with open(output_path, 'w') as f:
            f.writelines(minimal_requirements)
        
        print(f"Minimal requirements created: {len(all_requirements)} → {len(minimal_requirements)} packages")
        print(f"Saved to {output_path}")
        
        return output_path
    
    @staticmethod
    def create_docker_optimized(base_image: str = "python:3.9-slim") -> str:
        """Create optimized Dockerfile for deployment."""
        dockerfile_content = f"""# Optimized Dockerfile for Hate Speech Detection
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy optimized code
COPY optimized_hate_speech_detection.py .
COPY config.py .

# Copy pre-computed embeddings if available
COPY embeddings_compressed.npz ./data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_TRANSFORMERS_CACHE=/tmp/transformers_cache

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import optimized_hate_speech_detection; print('OK')" || exit 1

# Run the application
CMD ["python", "optimized_hate_speech_detection.py"]
"""
        
        with open("Dockerfile.optimized", "w") as f:
            f.write(dockerfile_content)
        
        print("Optimized Dockerfile created: Dockerfile.optimized")
        return "Dockerfile.optimized"


def run_all_optimizations():
    """Run all available optimizations."""
    print("=" * 60)
    print("RUNNING ALL OPTIMIZATIONS")
    print("=" * 60)
    
    # Initialize optimizers
    bundle_optimizer = BundleSizeOptimizer()
    memory_optimizer = MemoryOptimizer()
    deployment_optimizer = DeploymentOptimizer()
    
    # 1. Optimize images
    print("\n1. Optimizing images...")
    if os.path.exists("."):
        bundle_optimizer.optimize_images(".", "optimized_images")
    
    # 2. Clean notebook
    print("\n2. Cleaning Jupyter notebook...")
    if os.path.exists("Hate_Speech_Uncertainty_Estimation.ipynb"):
        bundle_optimizer.clean_jupyter_notebook(
            "Hate_Speech_Uncertainty_Estimation.ipynb",
            "Hate_Speech_Uncertainty_Estimation_cleaned.ipynb"
        )
    
    # 3. Create minimal requirements
    print("\n3. Creating minimal requirements...")
    used_packages = [
        'torch', 'transformers', 'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'tqdm', 'sentencepiece'
    ]
    deployment_optimizer.create_requirements_minimal(
        "requirements.txt", used_packages, "requirements_minimal.txt"
    )
    
    # 4. Create optimized Dockerfile
    print("\n4. Creating optimized Dockerfile...")
    deployment_optimizer.create_docker_optimized()
    
    print("\n" + "=" * 60)
    print("ALL OPTIMIZATIONS COMPLETED")
    print("=" * 60)
    print("\nFiles created:")
    print("- optimized_images/ (optimized PNG files)")
    print("- Hate_Speech_Uncertainty_Estimation_cleaned.ipynb (cleaned notebook)")
    print("- requirements_minimal.txt (minimal dependencies)")
    print("- Dockerfile.optimized (optimized Docker setup)")


if __name__ == "__main__":
    run_all_optimizations()