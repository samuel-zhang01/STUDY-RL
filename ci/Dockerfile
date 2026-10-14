# Dependency-only preparation: no application, model, Git config or owner state.
FROM node:22.22.2-bookworm@sha256:03746f6aafd73f2da621094d5e27426225820e93bc202daea0cf1fa388c1ce09 AS node
FROM python:3.12-slim-bookworm@sha256:2ed6491b93cd49272ee6de2b5a38440c3448360322c089fc23e370722d74179d
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
ENV PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PLAYWRIGHT_BROWSERS_PATH=/ms-playwright NPM_CONFIG_USERCONFIG=/dev/null NPM_CONFIG_GLOBALCONFIG=/tmp/npm-global-empty NPM_CONFIG_REGISTRY=https://registry.npmjs.org NPM_CONFIG_STRICT_SSL=true
RUN umask 022 && apt-get update && apt-get install -y --no-install-recommends make git build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* && touch /tmp/npm-global-empty
COPY --chmod=0644 requirements.txt requirements-llm.txt requirements-ci.txt /opt/requirements/
RUN python -m pip install --no-cache-dir 'swig==4.*' \
    && python -m pip install --no-cache-dir -r /opt/requirements/requirements-ci.txt \
    && python -m pip install --no-cache-dir -r /opt/requirements/requirements.txt -r /opt/requirements/requirements-llm.txt \
    && python -m pip check
WORKDIR /opt/ci-node
COPY --chmod=0644 package.json package-lock.json ./
RUN umask 022 && npm ci --ignore-scripts --include=dev --no-audit --no-fund \
    && npx --no-install playwright install --with-deps chromium firefox webkit \
    && chmod -R a+rX /ms-playwright
# Same upstream version/checksum as the original full-history workflow.
RUN curl --fail --silent --show-error --location --proto '=https' \
      https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz -o /tmp/gitleaks.tar.gz \
    && echo '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb  /tmp/gitleaks.tar.gz' | sha256sum --check \
    && tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks && rm /tmp/gitleaks.tar.gz
WORKDIR /tmp
USER 10001:10001
