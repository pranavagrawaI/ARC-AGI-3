"""API totals; inferred image-object counts never claim native image inspection."""
TOKENS_PER_IMAGE=400
LIMITATIONS=[
    'A matching aggregate cannot prove each image was consumed once or correspondence.',
    'No pixel-processing, tensor, cache-tensor or native processor equivalence established.',
    'Image-object count is inferred from aggregate tokens under pinned 400 tokens/image.',
]

def observe(arm,record,body):
    usage=body['usage'];details=usage['prompt_tokens_details']
    total=details['image_tokens'] if arm=='control' else details['multimodal_tokens']['image']
    prompt=usage['prompt_tokens'];completion=usage['completion_tokens'];count=len(record['images'])
    if type(total) is not int or total!=TOKENS_PER_IMAGE*count:raise ValueError('missing/mismatched complete image-token total')
    if type(prompt) is not int or prompt!=record['local_prompt']:raise ValueError('corrected prompt total mismatch/truncation')
    if type(completion) is not int or not 0<completion<=record['cap']:raise ValueError('completion cap/usage')
    if usage.get('total_tokens')!=prompt+completion:raise ValueError('inconsistent total usage')
    if len(body['choices'])!=1 or body['choices'][0]['finish_reason'] not in ('length','stop','tool_calls'):raise ValueError('finish reason')
    if body['choices'][0].get('message') is None:raise ValueError('missing response message')
    return dict(source_png_count=count,source_png_sha256=record['images'],observed_api_image_tokens=total,
                pinned_expected_tokens_per_image=TOKENS_PER_IMAGE,inferred_image_object_count=total//TOKENS_PER_IMAGE,
                corrected_prompt_total=prompt,completion_tokens=completion,
                cached_tokens=details.get('cached_tokens'),image_count_observation='INFERRED_AGGREGATE_ONLY')
