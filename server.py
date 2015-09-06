import argparse
import pyjsonrpc

from gensim.models.word2vec import Word2Vec

W2V_MODEL = None

class Word2VecRequestHandler(pyjsonrpc.HttpRequestHandler):
    """ a simple handler class that implements the 'expand' method """

    @pyjsonrpc.rpcmethod
    def expand(self, word):
        """ expand the word using the word2vec model """

        try:
            result = W2V_MODEL.most_similar(positive=[word], negative=[], topn=10)
        except:
            result = []
        return [pair[0] for pair in result]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A word2vec-based term-expansion JSON/RPC server')
    parser.add_argument('model', metavar='MODEL',
                        help='word2vec model in binary formmat')
    parser.add_argument('port', metavar='PORT', type=int,
                        help='port number')
    args = parser.parse_args()

    # Set up word2vec model
    W2V_MODEL = Word2Vec.load_word2vec_format(args.model, binary=True)
    W2V_MODEL.init_sims(replace=True)

    http_server = pyjsonrpc.ThreadingHttpServer(
        server_address=('localhost', args.port),
        RequestHandlerClass=Word2VecRequestHandler)
    print "Starting word2vec server ..."
    print "URL: http://localhost:{}".format(args.port)
    http_server.serve_forever()
