"""Read-only GetKernel audit. ListKernels omits currentVersionNumber.

Kept outside the frozen launcher; no candidate or qualification rule changes.
"""
import argparse
import json
from prepare import HERE, SLUG, digest, expected
from run import REF, api_setup, now, save, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--resolve-initial', action='store_true')
    args = parser.parse_args()
    verify()
    api, _ = api_setup()
    Request = api.kernels_pull.__func__.__globals__['ApiGetKernelRequest']
    request = Request()
    request.user_name, request.kernel_slug = REF.split('/')
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    metadata = json.loads(str(response.metadata))
    assert metadata['currentVersionNumber'] == 1
    assert metadata['id'] == 141908958 and metadata['ref'] == REF
    planned_meta = json.loads((HERE / 'kernel/kernel-metadata.json').read_text())
    for remote_key, local_key in {
        'isPrivate': 'is_private', 'enableGpu': 'enable_gpu',
        'enableInternet': 'enable_internet', 'machineShape': 'machine_shape',
        'dockerImage': 'docker_image', 'datasetDataSources': 'dataset_sources',
        'modelDataSources': 'model_sources', 'competitionDataSources': 'competition_sources',
    }.items():
        assert metadata[remote_key] == planned_meta[local_key], remote_key
    project = lambda nb: [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]
    remote = json.loads(response.blob.source)
    assert project(remote) == project(expected()), 'source mismatch'
    folder = HERE / args.label
    folder.mkdir(exist_ok=False)
    notebook = folder / (SLUG + '.ipynb')
    notebook.write_text(response.blob.source)
    save(folder / 'metadata.json', metadata)
    result = {'utc': now(), 'ref': REF, 'version': 1, 'all_cells_equal': True,
              'sha256': digest(notebook), 'status': str(api.kernels_status(REF).status),
              'verification_api': 'GetKernel metadata.currentVersionNumber and blob.source'}
    save(folder / 'identity.json', result)
    if args.resolve_initial:
        initial = HERE / 'source-readback' / (SLUG + '.ipynb')
        assert project(json.loads(initial.read_text())) == project(remote)
        assert not (initial.parent / 'identity.json').exists()
        save(initial.parent / 'identity.json', {**result, 'sha256': digest(initial),
             'independent_readback': str(folder.relative_to(HERE)),
             'resolution': 'ListKernels omitted version; GetKernel confirms latest version1. Original error retained.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
