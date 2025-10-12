# Builder stage: create a conda env with binary pyarrow and project deps
FROM condaforge/mambaforge:latest AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Create conda env and install binary packages
RUN mamba create -y -n xr_env python=3.11 pyarrow=21.0.0 numpy pip && \
    /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate xr_env && pip install --upgrade pip setuptools wheel uv"

# Copy project and install Python deps via uv into the conda env
COPY pyproject.toml .
COPY . .
# Install runtime Python packages directly into the conda env's pip so executables (streamlit) are placed in the env bin
RUN /bin/bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate xr_env && \
    pip install --upgrade pip setuptools wheel && \
    pip install streamlit==1.45.1 pydicom==3.0.1 matplotlib==3.10.3 opencv-python==4.11.0.86 scikit-image==0.25.2 'git+https://github.com/jlleongarcia/pylinac.git@7a0680efc944536eb9df550229571ced81e83de2'"

# (no conda-pack step; we'll copy the env directory to the runtime image)

# Runtime stage: slim runtime with the packed conda env unpacked
FROM python:3.11-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y libstdc++6 && rm -rf /var/lib/apt/lists/*

# Copy the full conda installation from the builder so the env prefix remains valid
COPY --from=builder /opt/conda /opt/conda
ENV PATH="/opt/conda/envs/xr_env/bin:${PATH}"

# Copy app code
COPY . .

EXPOSE 8502

ENTRYPOINT ["/bin/bash", "-lc", "/opt/conda/envs/xr_env/bin/streamlit run menu_analyzer.py --server.port=8502 --server.enableCORS=true --server.enableXsrfProtection=false"]
