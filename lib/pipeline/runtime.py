import os


def body_model_root(args):
    """SMPL/SMPL-X model directory: ``--body-model-root``, else the copy PromptHMR already requires."""
    return getattr(args, 'body_model_root', None) or args.prompthmr_root / 'data/body_models'


def runtime(args):
    """Create ``<output_root>/runtime`` with the asset links the components resolve relative to cwd, and enter it."""
    from lib.gvhmr import set_body_model_root
    set_body_model_root(body_model_root(args))
    workspace = args.output_root / 'runtime'
    workspace.mkdir(parents=True, exist_ok=True)
    links = {'data': args.asset_root, 'thirdparty/DROID-SLAM': args.droid_root}
    for name, target in links.items():
        link = workspace / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_dir():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f'Runtime asset path changed: {link}')
        elif link.exists():
            raise FileExistsError(link)
        else:
            try:
                link.symlink_to(target.resolve(), target_is_directory=True)
            except FileExistsError:
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise
    os.chdir(workspace)
