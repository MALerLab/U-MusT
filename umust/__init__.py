import importlib
import importlib.abc
import importlib.util
import sys

_LEGACY_PACKAGE_NAME = 'lsamt'


class _LegacyAliasLoader(importlib.abc.Loader):
  def create_module(self, spec):
    return importlib.import_module(spec.name.replace(_LEGACY_PACKAGE_NAME, __name__, 1))

  def exec_module(self, module):
    pass


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
  """Datasets preprocessed before the package rename contain pickled objects
  referencing `lsamt.*` module paths (e.g. note-event .npy files). Resolve
  those imports to the corresponding umust modules so the pickles load."""

  def find_spec(self, fullname, path=None, target=None):
    if fullname == _LEGACY_PACKAGE_NAME or fullname.startswith(_LEGACY_PACKAGE_NAME + '.'):
      return importlib.util.spec_from_loader(fullname, _LegacyAliasLoader())
    return None


sys.meta_path.append(_LegacyAliasFinder())
