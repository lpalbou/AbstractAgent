"""Manual real-model acceptance probe. Uses fictional local tools; no external actions.
Run with sibling sources on PYTHONPATH and an already-loaded LM Studio model.
Set AVATAR_TEST_MODEL and AVATAR_TEST_BASE_URL to target a different local model.
"""
import json,time,os
from abstractagent.agents.react import ReactAgent
from abstractcore.tools import tool
from abstractruntime.integrations.abstractcore import create_local_runtime,MappingToolExecutor
from abstractruntime.storage.steer_sidecar import InMemorySteerSidecar
from abstractruntime.core.models import RunStatus
seen=[]
@tool
def list_restaurants() -> str:
    """Read the test restaurant catalog, including accessibility and opening hours."""
    return json.dumps({'fixture':True,'places':[{'name':'Park Cafe','quiet':True,'wheelchair_access':False,'hours':'11:00-15:00'},{'name':'Riverside','quiet':False,'wheelchair_access':True,'hours':'11:30-15:30'}]})
@tool
def check_opening_hours(place: str) -> str:
    """Check the published lunch opening hours of a named test restaurant."""
    seen.append(place)
    return json.dumps({'fixture':True,'place':place,'open_for_lunch':True})
events=[]
def on_step(name,data):
    if name in ('public_update','public_update_rejected'):events.append((name,data));print(name,json.dumps(data),flush=True)
runtime=create_local_runtime(provider='lmstudio',model=os.environ.get('AVATAR_TEST_MODEL','qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive'),llm_kwargs={'base_url':os.environ.get('AVATAR_TEST_BASE_URL','http://127.0.0.1:1234/v1'),'timeout':90},steer_store=InMemorySteerSidecar(),tool_executor=MappingToolExecutor.from_tools([list_restaurants,check_opening_hours]))
agent=ReactAgent(runtime=runtime,tools=[list_restaurants,check_opening_hours],on_step=on_step,review_mode=False,max_iterations=8,max_output_tokens=1600,session_id='avatar-local-acceptance')
rid=agent.start('This is a test with fictional restaurants. Find a quiet place for lunch. First call list_restaurants, then tell me in a public update which option you are leaning toward and why. Then check opening hours using check_opening_hours before the final recommendation. Briefly communicate meaningful findings as you work. Do not ask questions.',conversation={'enabled':True},temperature=0)
steered=False;started=time.monotonic()
for i in range(80):
    state=agent.step()
    if not steered and any(n=='public_update' for n,d in events):
        msg='Please keep Riverside in the comparison. Wheelchair access is essential for my mother; a quiet atmosphere is only a preference. Clearly acknowledge this correction in your final recommendation.'
        runtime.steer(rid,msg,display_text=msg,speaker='person:test');steered=True;print('CORRECTION_ACCEPTED',flush=True)
    if state.status in (RunStatus.COMPLETED,RunStatus.FAILED,RunStatus.CANCELLED,RunStatus.WAITING):break
ledger=runtime.get_ledger(rid)
result={'status':state.status.value,'steered':steered,'seconds':round(time.monotonic()-started,2),'tool_checks':seen,'public_updates':[d for n,d in events if n=='public_update'],'answer':state.output.get('answer') if isinstance(state.output,dict) else state.output,'error':state.error,'public_human_receipts':sum(1 for r in ledger if (r.get('effect')or{}).get('payload',{}).get('name')=='user_guidance')}
open('/tmp/avatar-local-result.json','w').write(json.dumps(result,indent=2,default=str));print(json.dumps(result,default=str),flush=True)
assert state.status==RunStatus.COMPLETED,result
assert steered and result['public_updates'] and result['public_human_receipts']==1,result
assert 'Riverside' in result['answer'] and 'Riverside' in seen,result
