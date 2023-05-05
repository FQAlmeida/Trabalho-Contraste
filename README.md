# Trabalho Contraste

Esse trabalho foi desenvolvido durante no Semestre 01/2023, para a disciplina de Processamento de Imagens (PIM).

Universidade Estadual de Santa Catarina.

Autores:
- Andrei
- Otávio Almeida

Orientador:
- Gilmário B. Santos

## Instruções de Instalação

O projeto depende de duas ferramentas para ser instalado.

### Pyenv

Pyenv é um gerenciador de versões do Python.
Utilizaremos ele para garantir que a versão seja a 3.11.

#### Instalação

- Unix

```bash
curl https://pyenv.run | bash
```

- Windows

```PowerShell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

Reinicie o shell, e garanta que o pyenv está instalado.

```bash
pyenv --version
```

Se tudo estiver ok.

```bash
pyenv install 3.11.3
```

Reinicie novamente o shell, ative a versão do python,e garanta que o python está instalado, na versão correta.

```bash
pyenv global 3.11.3
pyenv local 3.11.3 # Na pasta do projeto

python --version # 3.11.3
```

### Poetry
Poetry é um gerenciador de dependências para python.
Utilizaremos ele para garantir que todas as dependências de terceiros estão instaladas corretamente.

#### Instalação

- Unix

```bash
curl -sSL https://install.python-poetry.org | python -
```

- Windows

```PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

Verifique que o poetry foi instalado.

```bash
poetry --version
```

Então, precisaremos instalar as dependêcias, e ativar o shell.

```bash
# Na pasta do projeto
poetry install

poetry shell
```

No shell do poetry, para abrir o dashboard.

```bash
task dashboard
```
