import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from .slam_utils import zoedepth_load_compat, zoedepth_midas_beit_compat


def local_resources():
    names = ['ZOEDEPTH_REPO', 'MIDAS_REPO', 'ZOEDEPTH_CHECKPOINT']
    values = [os.environ.get(name) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise ValueError('Offline depth loading requires ZOEDEPTH_REPO, MIDAS_REPO and ZOEDEPTH_CHECKPOINT')
    zoe, midas, checkpoint = [Path(value).resolve() for value in values]
    for path in [zoe / 'hubconf.py', zoe / 'zoedepth/models/builder.py',
                 midas / 'hubconf.py', checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    return zoe, midas, checkpoint


def load_zoedepth_model():
    resources = local_resources()
    with zoedepth_load_compat():
        if resources is None:
            model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True, trust_repo=True)
        else:
            zoe, midas, checkpoint = resources
            for name in ['zoedepth', 'midas']:
                loaded = sys.modules.get(name)
                root = zoe if name == 'zoedepth' else midas
                if loaded is not None:
                    locations = [loaded.__file__] if getattr(loaded, '__file__', None) else list(getattr(loaded, '__path__', []))
                    if not locations or not all(Path(p).resolve().is_relative_to(root) for p in locations):
                        raise RuntimeError(f'{name} already imported from a different source directory')
            previous_path = list(sys.path)
            hub_load = torch.hub.load
            def local_hub(repo, model, *args, **kwargs):
                if repo not in ('intel-isl/MiDaS', 'isl-org/MiDaS'):
                    raise RuntimeError(f'Unexpected Torch Hub dependency during offline load: {repo}')
                if kwargs.get('pretrained', False):
                    raise RuntimeError('MiDaS weights must come from the full local ZoeDepth checkpoint')
                kwargs.pop('force_reload', None)
                kwargs.pop('trust_repo', None)
                return hub_load(str(midas), model, *args, source='local', **kwargs)
            def no_download(*args, **kwargs):
                raise RuntimeError('Network download requested during offline ZoeDepth loading')
            try:
                sys.path.insert(0, str(zoe))
                get_config = importlib.import_module('zoedepth.utils.config').get_config
                build_model = importlib.import_module('zoedepth.models.builder').build_model
                with patch.object(torch.hub, 'load', side_effect=local_hub), patch.object(
                    torch.hub, 'load_state_dict_from_url', side_effect=no_download
                ), patch.object(torch.hub, 'download_url_to_file', side_effect=no_download):
                    config = get_config('zoedepth', 'infer',
                                        pretrained_resource='local::' + str(checkpoint),
                                        use_pretrained_midas=False)
                    model = build_model(config)
            finally:
                sys.path[:] = previous_path
    zoedepth_midas_beit_compat(model)
    return model.eval()
