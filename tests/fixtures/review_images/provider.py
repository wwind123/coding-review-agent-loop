"""Inert canned HTTP provider; used only inside an isolated network namespace."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

provider = None
requests=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')
 def do_POST(self):
  data=json.loads(self.rfile.read(int(self.headers['Content-Length']))); requests.append(data)
  if provider=='codex':
   events=[('response.created',{'response':{'id':'resp_test','object':'response','model':'provider-substitute','status':'in_progress','output':[]}}),('response.output_item.added',{'output_index':0,'item':{'id':'msg_test','type':'message','role':'assistant','content':[]}}),('response.output_text.delta',{'item_id':'msg_test','output_index':0,'content_index':0,'delta':'VISUAL_OK'}),('response.output_item.done',{'output_index':0,'item':{'id':'msg_test','type':'message','role':'assistant','content':[{'type':'output_text','text':'VISUAL_OK','annotations':[]}]}}),('response.completed',{'response':{'id':'resp_test','object':'response','model':'provider-substitute','status':'completed','output':[],'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}}})]
  else:
   model=data.get('model','claude-sonnet-4-6')
   events=[('message_start',{'type':'message_start','message':{'id':'msg_test','type':'message','role':'assistant','model':model,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':1,'output_tokens':0}}}),('content_block_start',{'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}}),('content_block_delta',{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'VISUAL_OK'}}),('content_block_stop',{'type':'content_block_stop','index':0}),('message_delta',{'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':1}}),('message_stop',{'type':'message_stop'})]
  self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
  for kind,event in events:
   if provider=='codex': event['type']=kind
   self.wfile.write(('event: '+kind+'\ndata: '+json.dumps(event)+'\n\n').encode()); self.wfile.flush()
