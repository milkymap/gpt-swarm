import zmq 
import zmq.asyncio as aiozmq 

from uuid import uuid4

import httpx 
import signal 
import asyncio
from openai import AsyncOpenAI

from time import time 
from libraries.log import logger 


from typing import List, Dict, Optional, Any, Tuple
from dataschema import Role, Message, REQUEST_TYPE

class GPTSwarm:
    def __init__(self, openai_api_key:str, nb_tokens_per_mn:int, nb_requests_per_mn:int, model_token_size:int):
        self.openai_api_key = openai_api_key
        self.nb_tokens_per_mn = nb_tokens_per_mn
        self.nb_requests_per_mn = nb_requests_per_mn

        self.model_token_size = model_token_size
        self.period = 1 / float(self.nb_requests_per_mn / 60)  # 0.02s for 3000 reqs

        self.collector_address = 'inproc://collector_endpoint'
        self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
    def stop_swarm(self):
        logger.info('stop-swarming was called ...!')
        running_task = asyncio.all_tasks(self.loop)
        for task in running_task:
            if task.get_name().startswith('worker-'):
                task.cancel()

    async def collector(self):
        pull_socket:aiozmq.Socket = self.ctx.socket(zmq.PULL) 
        pull_socket.bind(self.collector_address)

        try:
            start = time()
            keep_loop = True 
            while keep_loop:
                socket_status = await pull_socket.poll(timeout=100)
                if socket_status == zmq.POLLIN:
                    _ = await pull_socket.recv()  # consume the message 
                    if not self.start_timer.is_set():
                        self.start_timer.set()
                        start = time()

                    async with self.mutex:
                       logger.debug(f'collector is running {self.total_tokens}')
                       if self.consumed_tokens >= self.nb_tokens_per_mn - 2 * self.model_token_size:
                           self.tokens_status.clear()
                           logger.debug(f'TPM was reached => collector send the stop signal to all workers')
                       
                if self.start_timer.is_set():
                    end = time()
                    duration = end - start 
                    logger.debug(f'collector will reset the TPM in {(60 - duration):07.3f} seconds')
                
                    if duration > 60:
                        async with self.mutex:
                            self.nb_requests = 0 
                            self.consumed_tokens = 0
                            self.tokens_status.set()  # allow worker to make request
                            logger.debug('collector has set the timer')
                        self.start_timer.clear()  # wait a new response to start the timer 

            # end while loop 
        except asyncio.CancelledError:
            logger.warning('collector was cancelled')
        except Exception as e:
            logger.error(e)
        
        pull_socket.close(linger=0)
        logger.debug('collector has released its ressources')

    async def worker_strategy(self, push_socket:aiozmq.Socket, response, error=None) -> Tuple[Optional[Message], bool]:
        outgoing_message, keep_loop = None, False    
        try:
            if error is None and response:  # successful response 
                # Extract tokens from the response usage
                consumed_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else self.model_token_size
                async with self.mutex:
                    self.total_tokens = self.total_tokens + consumed_tokens
                    self.consumed_tokens = self.consumed_tokens + consumed_tokens
                
                # Create message from response output
                outgoing_message = Message(role=Role.ASSISTANT, content=response.output_text)
                push_socket.send(b'...')  # tell the collector that we have a new response 
                keep_loop = False             
            elif error:
                # Handle different error types
                if "401" in str(error) or "unauthorized" in str(error).lower():  # authorization failed 
                    keep_loop = False
                elif "429" in str(error) or "500" in str(error) or "rate limit" in str(error).lower():  # rate limit or server error
                    keep_loop = True 
                else:
                    keep_loop = False 
            else:
                keep_loop = False 
        except Exception as e:
            logger.error(e)
        return outgoing_message, keep_loop

    async def worker(self, worker_id:str, nb_retries:int, messages:List[Message]) -> Optional[Message]:
        push_socket:aiozmq.Socket = self.ctx.socket(zmq.PUSH)
        push_socket.connect(self.collector_address)

        delay = 0
        counter = 0
        keep_loop = True
        outgoing_message = None  
        
        while keep_loop:
            try:
                async with self.mutex:
                    self.nb_requests = self.nb_requests + 1 
                    delay = self.period * (self.nb_requests - 1)
                
                await asyncio.sleep(delay)
                logger.debug(f'{worker_id} is running >> nb_retries : {counter} and is waiting the TPM signal')
                
                async with self.mutex:
                    tpm_signal = self.tokens_status.is_set()
                
                if tpm_signal:
                    logger.debug(f'{worker_id} has received the TPM signal')
                    try:
                        # Convert messages to the format expected by Responses API
                        input_messages = []
                        for msg in messages:
                            input_messages.append({
                                "role": msg.role.value,
                                "content": msg.content
                            })
                        
                        # Use OpenAI Responses API with GPT-4o model
                        response = await self.openai_client.responses.create(
                            model="gpt-4o",
                            input=input_messages,
                            timeout=10
                        )
                        outgoing_message, keep_loop = await self.worker_strategy(push_socket, response)
                        counter = counter + int(keep_loop)  # 1 or 0 depends on the status of keep loop
                    except Exception as api_error:
                        logger.warning(f'{worker_id} API error: {api_error}')
                        outgoing_message, keep_loop = await self.worker_strategy(push_socket, None, api_error)
                        counter = counter + int(keep_loop)
                    
                else:
                    await self.tokens_status.wait()

            except asyncio.TimeoutError:
                logger.warning(f'{worker_id} has timeouted')
                counter = counter + 1  
            except asyncio.CancelledError:
                logger.warning(f'{worker_id} was cancelled')
                keep_loop = False 
            except Exception as e:
                logger.error(e)
                keep_loop = False 
            keep_loop = keep_loop and counter < nb_retries
        # end while loop ...!
        push_socket.close(linger=0) 
        logger.debug(f'{worker_id} has released its ressources')
        return outgoing_message

    async def swarm(self, conversations:List[List[Message]]) -> Optional[List[Message]]:
        self.nb_requests = 0 
        self.total_tokens = 0 
        self.consumed_tokens = 0 
        self.tokens_status.set()  # consumed_tokens == 0 => let workers start making requests 

        collector_task = asyncio.create_task(
            coro=self.collector()
        )

        worker_responses = None 
        try:
            awaitables = []
            for messages in conversations:
                worker_id = str(uuid4())
                task = asyncio.create_task(
                    coro=self.worker(
                        worker_id=worker_id, 
                        nb_retries=3,
                        messages=messages
                    ),
                    name=f'worker-{worker_id}'
                )
                awaitables.append(task)
            
            worker_responses = await asyncio.gather(*awaitables, return_exceptions=False)
        except asyncio.CancelledError:
            logger.debug('swarm was cancelled...!')
        except Exception as e:
            logger.error(e)
        
        collector_task.cancel()
        await collector_task
        return worker_responses     

    async def __aenter__(self):
        self.mutex = asyncio.Lock()
        self.tokens_status = asyncio.Event()
        self.start_timer = asyncio.Event()

        self.ctx = aiozmq.Context()
        self.loop = asyncio.get_running_loop()
        self.loop.add_signal_handler(
            sig=signal.SIGINT,
            callback=self.stop_swarm 
        )
        logger.debug('swarm was initalized')
        return self 
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.warning(f'Exception => {exc_value}')
            logger.exception(traceback)
        if self.loop.is_running():
            self.loop.stop()

        self.ctx.term()
        logger.debug('swarm was released')



