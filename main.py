import os
from document_analyzer.document_analyzer import DocumentAnalyzer

DIRECTORY = 'img/PRImA Layout Analysis Dataset/Images'


def main(path=None, debug=False):
    if path is not None:
        analyze_document(path, debug)
        return

    for filename in os.listdir(DIRECTORY):
        if filename.endswith('.tif'):
            analyze_document(DIRECTORY + '/' + filename, debug)


def analyze_document(path, debug):
    document_analyzer = DocumentAnalyzer(path, debug)
    document_analyzer.analyze_document()


if __name__ == '__main__':
    # args = sys.argv[1:]
    # if len(args) == 0:
    #     print('Argument missing. Taking argument from input:')
    #     args.append(input())
    # elif len(args) > 1:
    #     raise Exception(f'Too many arguments: {len(args)}. Only one argument is required.')

    # main(args[0])
    # main('img/la.png')
    # main('img/PRImA Layout Analysis Dataset/Images/00000880.tif', True)
    main()
