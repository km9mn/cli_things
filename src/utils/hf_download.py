from huggingface_hub import snapshot_download
import argparse

def download(args):
    snapshot_download(repo_id=args.repo_id,
                      local_dir=args.local_dir,
                      repo_type=args.repo_type,
                      local_dir_use_symlinks=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--repo-id', type=str)
    parser.add_argument('--local-dir', type=str)
    parser.add_argument('--repo-type', type=str, default=None)

    args = parser.parse_args()

    print(args)

    download(args)

