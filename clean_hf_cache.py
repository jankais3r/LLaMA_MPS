#!/usr/bin/env python3

from huggingface_hub import scan_cache_dir

hf_cache_info = scan_cache_dir()
for repo in list(hf_cache_info.repos):
	if repo.repo_id in ('tloen/alpaca-lora-7b', 'decapoda-research/llama-7b-hf'):
		for rev in list(repo.revisions):
			delete_strategy = scan_cache_dir().delete_revisions(rev.commit_hash)
			print(f'Deleting cache for repository {repo.repo_id} ({delete_strategy.expected_freed_size_str}).')
			delete_strategy.execute()
