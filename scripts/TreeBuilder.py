class TreeBuilder(object):
    """docstring for TreeBuilder"""
    def __init__(self, src_file="dictionary.json"):
        super(TreeBuilder, self).__init__()
        self.src_file = src_file


    def load_src(self):
        with open(self.src_file,"r"):

        