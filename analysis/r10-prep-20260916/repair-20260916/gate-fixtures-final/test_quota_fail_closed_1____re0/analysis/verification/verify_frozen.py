"""Read-only exact frozen bytes check for the independent launch reviewer. No approval."""
import argparse
import json
import launch_control as gate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args(argv)
    try:
        record=gate.validate(args.run_dir,launch_ready=False)
        print(json.dumps({'status':'FROZEN_BYTES_VERIFIED_NOT_LAUNCH_APPROVAL','manifest_sha256':gate.sha(gate.RUN_DIR/'manifest.json'),'sources':len(record['sources']),'provider_calls':0}))
        return 0
    except (OSError,ValueError,TypeError,KeyError) as error:
        print(json.dumps({'status':'HOLD','reason':str(error),'provider_calls':0}));return 2
if __name__=='__main__':raise SystemExit(main())
