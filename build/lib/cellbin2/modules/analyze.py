# Clustering and Annotation Module: facilitates HTML report generation

class Analyze(object):
    """ clustering, annotation part """
    # TODO:zhangying
    def __init__(self, data_path: str):
        self.data = None

    def cluster(self, ): pass

    def annotate(self, ): pass


def main():
    data_path = r'xxx.gef'
    aly = Analyze(data_path)
    aly.cluster()
    aly.annotate()


if __name__ == '__main__':
    main()
