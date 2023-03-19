import click 
import asyncio 

from libraries.swarming import GPTSwarm
from dataschema import Message, Role

async def run_model(openai_api_key:str):
    async with GPTSwarm(openai_api_key=openai_api_key, nb_tokens_per_mn=180_000, nb_requests_per_mn=3000, model_token_size=4096) as model:
        swarm_response = await model.swarm(
            conversations=[ 
                [Message(role=Role.USER, content='Please explain me the big bang in simple terms')] 
                for _ in range(32)  # 32 conversations in parallel
            ]
        ) 
        if swarm_response:
            for rsp in swarm_response:
                print(rsp)

@click.group(chain=False, invoke_without_command=False)
@click.option('--openai_api_key', envvar='OPENAI_API_KEY', help='openai api key for gpt-3.5', type=str, required=True)
@click.pass_context
def group(ctx:click.core.Context, openai_api_key:str):
    ctx.ensure_object(dict)
    ctx.obj['openai_api_key'] = openai_api_key

@group.command()
@click.pass_context
def start_swarming(ctx:click.core.Context):
    asyncio.run(run_model(ctx.obj['openai_api_key']))

if __name__ == '__main__':
    group(obj={})
