# torch-cuda-template
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/9a56f466d21f4dbf8de677fd5b8709d3)](https://www.codacy.com/manual/frgfm/torch-cuda-template?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/torch-cuda-template&amp;utm_campaign=Badge_Grade)![Build Status](https://github.com/frgfm/torch-cuda-template/workflows/python-package/badge.svg) [![codecov](https://codecov.io/gh/frgfm/torch-cda-template/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/torch-cuda-template) 



Template for CUDA / C++ extension writing with PyTorch



## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package using the default method:

```bash
pip install -e . --upgrade
```

which will build the CUDA extension and bind it with your python module.



## Usage

This python package can be used like any other:

```python
import torch
import cuda_ext.nn as nn

my_module = nn.DSigmoid()

with torch.no_grad():
	out = my_module(torch.rand(2, 3, 32, 32).cuda())
```



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## Credits

The content of this repo was brought together by the repo owner but highly benefited from the following resources:

- CUDA extensions of [Thomas Brandon](https://github.com/thomasbrandon)
- Official PyTorch [documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html)



## License

Distributed under the MIT License. See `LICENSE` for more information.