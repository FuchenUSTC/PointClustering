import model.pointnetpp as pointnetpp
import model.res16unet as res16unet
import model.point_transformer as point_transformer

MODELS = []

# load all models
def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])


add_models(pointnetpp)
add_models(res16unet)
add_models(point_transformer)


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]
  return NetClass
