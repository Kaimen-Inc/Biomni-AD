FROM mambaorg/micromamba:1.5.10

WORKDIR /app

ARG BIOMNI_ENV_FILE=biomni_env/environment.yml

COPY ${BIOMNI_ENV_FILE} /tmp/biomni_env.yml

RUN micromamba create -y -n biomni_e1 -f /tmp/biomni_env.yml && \
    micromamba run -n biomni_e1 pip install --no-cache-dir --upgrade pip && \
    micromamba run -n biomni_e1 pip install --no-cache-dir "chainlit>=1.0" && \
    micromamba clean --all --yes

COPY pyproject.toml README.md MANIFEST.in /app/
COPY biomni /app/biomni
COPY chainlit_app.py chainlit.md /app/
COPY .chainlit /app/.chainlit
COPY public /app/public
COPY docker/entrypoint.sh /app/docker/entrypoint.sh

RUN chmod +x /app/docker/entrypoint.sh && \
    micromamba run -n biomni_e1 pip install --no-cache-dir -e /app

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000 \
    BIOMNI_PATH=/app/data

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]