import os

import nbformat
from nbformat.v4.nbbase import new_markdown_cell

from generate_contents import iter_notebooks, NOTEBOOK_DIR


BOOK_COMMENT = "<!--BOOK_INFORMATION-->"


BOOK_INFO = BOOK_COMMENT + """
<a href="https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv" target="_blank"><img align="left" src="data/cover.jpg" style="width: 76px; height: 100px; background: white; padding: 1px; border: 1px solid black; margin-right:10px;"></a>
*This notebook contains an excerpt from the book [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv) by Michael Beyeler.
The code is released under the [MIT license](https://opensource.org/licenses/MIT),
and is available on [GitHub](https://github.com/mbeyeler/opencv-machine-learning).*

*Note that this excerpt contains only the raw code - the book is rich with additional explanations and illustrations.
If you find this content useful, please consider supporting the work by
[buying the book](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv)!*"""


def add_book_info():
    for nb_name in iter_notebooks():
        nb_file = os.path.join(NOTEBOOK_DIR, nb_name)
        nb = nbformat.read(nb_file, as_version=4)

        is_comment = lambda cell: cell.source.startswith(BOOK_COMMENT)

        if is_comment(nb.cells[0]):
            print('- amending comment for {0}'.format(nb_name))
            nb.cells[0].source = BOOK_INFO
        else:
            print('- inserting comment for {0}'.format(nb_name))
            nb.cells.insert(0, new_markdown_cell(BOOK_INFO))
        nbformat.write(nb, nb_file)


if __name__ == '__main__':
    add_book_info()
