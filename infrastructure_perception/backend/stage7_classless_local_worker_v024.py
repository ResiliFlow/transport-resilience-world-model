"""Portable isolated D130/ROI191 worker; learned extent, no provider calls."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,math,os,re,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from infrastructure_perception.backend import stage7_classless_worker_v022 as base
torch,np,F,Image=base.torch,base.np,base.F,base.Image
PACKAGE=ROOT/'infrastructure_perception/models/classless_local_v024'
def refine_box(box,delta):
    a=np.asarray(box,float);d=np.clip(np.asarray(delta,float),-2,2);size=a[2:]-a[:2]
    centre=(a[:2]+a[2:])/2+d[:2]*size;wh=np.exp(d[2:])*size
    b=np.r_[centre-wh/2,centre+wh/2];c=(a[:2]+a[2:])/2
    low=np.maximum(0,c-.9*size);high=np.minimum(1,c+.9*size)
    b[:2]=np.maximum(b[:2],low);b[2:]=np.minimum(b[2:],high)
    if not np.isfinite(b).all() or not (b[2:]>b[:2]).all():return list(box)
    return b.tolist()
class SupportedRegionModel(base.RegionModel):
    def __init__(self,weights):
        super().__init__(weights)
        self.extent=torch.nn.Linear(1536,4)
        self.extent.load_state_dict(weights['extent'],strict=True)
        self.trace=[]
    def raw_forward(self,pixels):
        with torch.autocast('cuda',dtype=torch.float16):values=self.vision(pixel_values=pixels).pooler_output
        features=F.normalize(values.float(),dim=1).reshape(len(pixels)//2,1536)
        return dict(logits=self.classes(features),hosts=self.hosts(features),deltas=self.extent(features),features=features)
    def forward(self,pixels):
        output=self.raw_forward(pixels);support=output['hosts'].sigmoid().max(1).values;admitted=support>=.2
        foreground=torch.logsumexp(output['logits'][:,:8],1)-output['logits'][:,8]
        self.trace.extend(dict(foreground_logit=float(z),support=float(s),admitted=bool(a)) for z,s,a in zip(foreground.cpu(),support.cpu(),admitted.cpu(),strict=True))
        logits=output['logits'].clone();logits[~admitted,:8]=-60;logits[~admitted,8]=0
        return dict(output,logits=logits)
class Engine:
    def __init__(self,manifest_sha,package=None):
        started=time.perf_counter();package=Path(package or PACKAGE).resolve()
        if not package.is_relative_to((ROOT/'infrastructure_perception/models').resolve()):raise ValueError('Package must remain inside the model directory')
        if not re.fullmatch('[A-F0-9]{64}',manifest_sha) or base.sha(package/'manifest.json')!=manifest_sha:raise ValueError('Local manifest mismatch')
        self.manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'));self.manifest_sha256=manifest_sha
        if base.sha(base.__file__)!=self.manifest['frozen_base_worker_sha256']:raise ValueError('Frozen shared inference changed')
        if base.sha(__file__)!=self.manifest['local_worker_sha256']:raise ValueError('Local implementation mismatch')
        for name,digest in self.manifest['files'].items():
            if Path(name).name!=name or base.sha(package/name)!=digest:raise ValueError('Local package asset mismatch')
        if self.manifest['local_policy']!={'foreground_logit':1.0096738338470461,'max_area_fraction':.25,'nms_iou':.5,'max_boxes':30,'support_floor':.2}:raise ValueError('Local selection policy mismatch')
        if not torch.cuda.is_available():raise RuntimeError('Compatible NVIDIA CUDA hardware/driver required')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        dense=torch.load(package/'dense.pt',map_location='cpu',weights_only=False)
        self.dense=dense['model'].float().eval().cuda().requires_grad_(False);self.dense.model[-1].max_det=300
        self.parameter_sha256=base.tensor_hash(self.dense)
        if self.parameter_sha256!=self.manifest['dense_parameter_sha256']:raise ValueError('Dense130 parameters mismatch')
        self.roi=SupportedRegionModel(torch.load(package/'roi.pt',map_location='cpu',weights_only=False)).eval().cuda().requires_grad_(False)
        if base.tensor_hash(self.roi)!=self.manifest['roi_export_parameter_sha256']:raise ValueError('ROI191 parameters mismatch')
        self.execution_count=0;self.load_seconds=time.perf_counter()-started
    @torch.inference_mode()
    def infer(self,request):
        if request.get('mode')!='local':raise ValueError('This worker is Local only')
        op,npth=Path(request['original']),Path(request['native']);osh,nsh=base.sha(op),base.sha(npth)
        if osh!=request['original_sha256'] or nsh!=request['native_sha256']:raise ValueError('Current frame identity changed')
        started=time.perf_counter();run_id='d130-r191-worker-'+uuid.uuid4().hex
        with Image.open(op) as raw:original=raw.convert('RGB')
        with Image.open(npth) as raw:native=raw.convert('RGB')
        self.roi.trace=[]
        try:
            tensor,mapping=base.tensor_views(original,native);torch.cuda.synchronize();dense_start=time.perf_counter()
            output=self.dense(tensor);raw=output[1] if isinstance(output,tuple) else output;branch=raw['one2many']
            boxes=self.dense.model[-1]._get_decode_boxes(branch).transpose(1,2)
            proposals=base.decode(boxes,branch['scores'],mapping,run_id);torch.cuda.synchronize();dense_seconds=time.perf_counter()-dense_start
            del tensor,output,raw,branch,boxes
            roi_start=time.perf_counter();zs=[];hosts=[];deltas=[]
            for offset in range(0,len(proposals),8):
                rows=proposals[offset:offset+8]
                pixels=torch.stack([base.crop_tensor(native,p['box'],factor) for p in rows for factor in (1.,1.8)]).cuda()
                output=self.roi(pixels);zs.extend((torch.logsumexp(output['logits'][:,:8],1)-output['logits'][:,8]).cpu().tolist())
                hosts.extend(output['hosts'].float().cpu().tolist());deltas.extend(output['deltas'].float().cpu().tolist());del pixels,output
            candidates=[]
            for p,z,host,delta in zip(proposals,zs,hosts,deltas,strict=True):
                b=refine_box(p['box'],delta)
                if z>=1.0096738338470461 and (b[2]-b[0])*(b[3]-b[1])<=.25:
                    candidates.append(dict(p,box=b,source_box=p['box'],extent_delta=delta,proposal_confidence=p['confidence'],foreground_logit=z,
                        confidence=1/(1+math.exp(-max(-60,min(60,z)))),surface_type=('road','kerb','footpath')[int(np.argmax(host))],host_logits=host,host_estimate_uncalibrated=True))
            predictions=[]
            if candidates:
                keep=base.nms(torch.tensor([p['box'] for p in candidates]),torch.tensor([p['foreground_logit'] for p in candidates]),.5)[:30].tolist()
                predictions=[candidates[i] for i in keep]
            torch.cuda.synchronize();self.execution_count+=1
            return dict(request_id=request['request_id'],run_id=run_id,worker_pid=os.getpid(),execution_count=self.execution_count,actual_model_execution=True,
                checkpoint_sha256=self.manifest['dense_source_sha256'],parameter_sha256=self.parameter_sha256,model_load_seconds=self.load_seconds,
                original_sha256=osh,native_sha256=nsh,dense_seconds=dense_seconds,roi_seconds=time.perf_counter()-roi_start,seconds=time.perf_counter()-started,
                forward_images=len(mapping),roi_execution=True,proposals=proposals,predictions=predictions,cross_frame_cache=False,
                gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_peak_bytes=torch.cuda.max_memory_allocated(),provider_calls=0,
                model_version=self.manifest['package_version'],manifest_sha256=self.manifest_sha256,surface_gate_trace=self.roi.trace,surface_gate_threshold=.2,
                roi_checkpoint_sha256=self.manifest['roi_source_sha256'],roi_parameter_sha256=self.manifest['roi_parameter_sha256'],roi_export_parameter_sha256=self.manifest['roi_export_parameter_sha256'])
        finally:original.close();native.close()
def main():
    cli=argparse.ArgumentParser();cli.add_argument('--manifest-sha',required=True);cli.add_argument('--package',type=Path);args=cli.parse_args()
    with contextlib.redirect_stdout(sys.stderr):engine=Engine(args.manifest_sha,args.package)
    print(json.dumps(dict(ready=True,pid=os.getpid(),manifest_sha256=args.manifest_sha,parameter_sha256=engine.parameter_sha256,model_version=engine.manifest['package_version'],load_seconds=engine.load_seconds)),flush=True)
    for line in sys.stdin:
        request={}
        try:
            request=json.loads(line)
            if request.get('command')=='shutdown':break
            with contextlib.redirect_stdout(sys.stderr):result=engine.infer(request)
            print(json.dumps(dict(ok=True,**result)),flush=True)
        except Exception as error:print(json.dumps(dict(ok=False,request_id=request.get('request_id'),error_type=type(error).__name__,error='Local inference failed; '+type(error).__name__)),flush=True)
if __name__=='__main__':main()
