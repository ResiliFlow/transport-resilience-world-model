'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v,d=1) => Number(v).toLocaleString('en-GB',{minimumFractionDigits:d,maximumFractionDigits:d});
const integer = v => num(v,0);
const roadCount = value => integer(value)+(value===1?' road unit':' road units');
const labels = {fatal:'Expected deaths',serious:'Expected serious injuries',high:'Chance of high harm',rising:'Chance of worsening harm'};
const optionLabels = {fatal:'Deaths',serious:'Serious injuries',high:'High harm (%)',rising:'Worsening harm (%)'};
const outcomeNames = {fatal:'Deaths',serious:'Serious injuries',high:'High harm',rising:'Worsening harm'};
const isRisk = metric => ['high','rising'].includes(metric);
const observedIndex = {fatal:8,serious:9,high:10,rising:11};
const actualLabel = (value,metric) => isRisk(metric)?(value?'Yes':'No'):integer(value);
const metricGroups = {people:{label:'Number of people',metrics:['serious','fatal']},risk:{label:'Risk',metrics:['high','rising']}};
const filterLabels = {future:'Show',history:'Check'};
const rankTitles = {fatal:'Most expected deaths',serious:'Most expected serious injuries',high:'Highest harm risk',rising:'Highest risk of worsening harm'};
const colours=['#43879b','#1c6983','#cfac27','#ce702f','#af3545'];
let state={mode:null,meta:null,query:null,result:null,selected:null,selectedData:null,selectionRequest:0,page:0,request:0,chartLimit:10,map:null,mapReady:false,geo:{},currentGeo:null,markers:[],streetErrors:0,streetLoaded:false};
let toastTimer;

async function api(url,body){
  const response=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
  if(!response.ok){const data=await response.json().catch(()=>({error:'The local service could not respond.'}));throw Error(data.error || 'Request failed.');}
  return response.json();
}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500);}
function error(message){$('status').classList.remove('ready');$('status').classList.add('error');$('status').textContent=message;}
function bind(id,fn,event='click'){$(id).addEventListener(event,e=>Promise.resolve(fn(e)).catch(err=>error(err.message)));}
function periodText(p){return p.replace('-','–');}
function formQuery(){return {dataset:state.mode==='history'?'history':'future',region:$('region').value,period:$('period').value,metric:$('metric').value,search:$('search').value.trim()};}
function score(v,metric=state.query?.metric){return ['high','rising'].includes(metric)?num(v*100,1)+'%':num(v,2);}

function clearResults(){
  RoadEngineering.clear();
  state.request++;state.selectionRequest++;state.query=null;state.result=null;state.selected=null;state.selectedData=null;state.clickPoint=null;state.currentGeo=null;state.page=0;
  document.body.dataset.results='empty';
  for(const id of ['stats','results-card','detail','map-selection','map-bottom','export-results'])$(id).hidden=true;
  $('export-results').removeAttribute('href');
  for(const id of ['stats','results-chart','accuracy-summary','road-table','detail','map-selection','legend','map-note'])$(id).innerHTML='';
  $('map-title').textContent='Road network';$('map-subtitle').textContent='';$('reset-map').textContent='Reset map';
  $('status').textContent='';$('status').className='status ready';$('apply').disabled=false;
  if(state.mapReady){
    state.map.getSource('roads').setData({type:'FeatureCollection',features:[]});
    for(const name of ['road-halo','road-lines','road-hit','selected-road','selected-centre'])state.map.setLayoutProperty(name,'visibility','none');
    state.tooltip?.remove();state.map.getCanvas().style.cursor='';highlight();
    requestAnimationFrame(()=>state.map.resize());
  }
}
async function configure(mode,preset=null){
  if(!state.meta||!['future','history'].includes(mode))return;
  const previous=$('region').value;
  clearResults();state.mode=mode;
  document.querySelectorAll('.view-tabs [data-mode]').forEach(b=>{const active=b.dataset.mode===mode;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  const history=mode==='history',data=state.meta.datasets[history?'history':'future'];
  $('road-suggestions').innerHTML=(data.road_names||[]).map(name=>`<option value="${esc(name)}"></option>`).join('');
  $('region').innerHTML=['All areas',...data.regions].map(r=>`<option>${esc(r)}</option>`).join('');
  const region=preset?.region ?? previous;
  if([...$('region').options].some(o=>o.value===region))$('region').value=region;
  const periods=history?data.periods.slice().reverse():['2027-2030','2027','2028','2029','2030','2027-2028','2027-2029'];
  $('period').innerHTML=periods.map(p=>`<option value="${p}">${periodText(p)}${p.includes('-')?' · total':''}</option>`).join('');
  const groups=['people','risk'];
  const metrics=groups.flatMap(group=>metricGroups[group].metrics);
  const option=m=>`<option value="${m}">${optionLabels[m]}</option>`;
  $('metric').innerHTML=groups.map(group=>`<optgroup label="${metricGroups[group].label}">${metricGroups[group].metrics.map(option).join('')}</optgroup>`).join('');
  $('metric-label').textContent=filterLabels[mode];
  $('period-label').textContent=history?'Year':'Forecast period';
  if(preset){if(periods.includes(preset.period))$('period').value=preset.period;if(metrics.includes(preset.metric))$('metric').value=preset.metric;$('search').value=preset.search||'';}
  const titles={future:'Explore future risk',history:'Check predictions'};
  $('control-title').textContent=titles[mode];
  document.body.dataset.mode=mode;
  $('controls').hidden=false;$('table-disclosure').open=false;$('table-disclosure').hidden=history;
  if(state.mapReady)requestAnimationFrame(()=>state.map.resize());
}

async function run(){
  if(!state.mode)return;
  const q=formQuery();
  clearResults();const request=state.request;$('apply').disabled=true;$('status').classList.remove('error','ready');$('status').textContent='Loading results…';
  try{
    const result=await api('/api/results?'+new URLSearchParams(q));
    if(request!==state.request)return;
    state.query=q;state.result=result;state.page=0;
    document.body.dataset.results='ready';for(const id of ['stats','results-card','detail','map-selection','map-bottom'])$(id).hidden=false;
    $('reset-map').textContent='Show all results';
    paintStats();paintList();paintCharts();
    $('status').textContent='Results updated.';$('status').classList.add('ready');
    $('map-title').textContent=(q.region==='All areas'?'All areas':q.region)+' · '+(q.dataset==='history'?'prediction check':'future risk');
    $('map-subtitle').textContent=periodText(q.period)+' · '+labels[q.metric]+(isRisk(q.metric)?'':' · people');
    await updateMap();
    if(request!==state.request)return;
    const first=result.rows[0];
    if(first)await selectRoad(first.id,Boolean(q.search));else {$('detail').innerHTML='<p class="eyebrow">ROAD DETAILS</p><h2>No matching roads</h2><p class="muted">Change the area or road search and try again.</p>';state.selected=null;state.selectedData=null;highlight();paintMapSelection();}
  }catch(err){if(request===state.request)error(err.message);}finally{if(request===state.request)$('apply').disabled=false;}
}

function stat(label,value,note){return `<div class="stat"><span class="stat-label">${esc(label)}</span><strong>${value}</strong><span class="stat-note">${esc(note)}</span></div>`;}
function paintStats(){
  const s=state.result.summary,q=state.query;
  if(q.dataset==='future')$('stats').innerHTML=stat('Expected deaths',num(s.expected_first),periodText(q.period))+stat('Expected serious injuries',num(s.expected_second),periodText(q.period))+stat('Road units',integer(s.count),num(s.km)+' km');
  else{
    const e=s.evaluation;
    const show=(value,d=2)=>value===null?'—':num(value,d);
    if(isRisk(q.metric)){
      $('stats').innerHTML=stat('Average predicted chance',e.n?num(e.predicted_mean*100)+'%':'—',roadCount(e.n)+' checked')+stat('Actually occurred',e.n?num(e.observed_mean*100)+'%':'—',integer(e.observed_total)+' of '+integer(e.n)+' road units')+stat('Probability error',show(e.brier,3),'0 is best · all checked roads');
    }else{
      const name=q.metric==='fatal'?'deaths':'serious injuries';
      $('stats').innerHTML=stat('Predicted '+name,num(e.predicted_total),q.period)+stat('Actual '+name,integer(e.observed_total),q.period)+stat('Average error per road',show(e.mae),'People · '+roadCount(e.n));
    }
  }
}
function listRows(){return state.result.rows;}
function paintList(){
  const rows=listRows(),q=state.query,pageSize=10,totalPages=Math.ceil(rows.length/pageSize);
  state.page=Math.max(0,Math.min(state.page,totalPages-1));
  const start=state.page*pageSize,shown=rows.slice(start,start+pageSize);
  $('list-title').textContent=q.dataset==='history'?'Predicted vs actual':rankTitles[q.metric];
  $('list-description').textContent=labels[q.metric]+' · '+periodText(q.period);
  $('list-count').textContent=roadCount(state.result.summary.count);
  $('export-results').hidden=!state.result.summary.count;
  $('export-results').href='/api/results.csv?'+new URLSearchParams(q);
  $('value-title').textContent=labels[q.metric];
  const history=q.dataset==='history';$('actual-title').hidden=true;$('difference-title').hidden=true;$('table-disclosure').hidden=history;
  $('table-toggle').textContent='Show road list'+(rows.length<state.result.summary.count?' · top '+integer(rows.length):'');
  if(history){$('road-table').innerHTML='';return;}
  $('road-table').innerHTML=shown.length?shown.map(r=>`<tr class="${r.id===state.selected?'selected':''}" data-road-id="${esc(r.id)}"><td><button class="text-link" data-road="${esc(r.id)}">${esc(r.road)}</button><small>${esc(r.id)}</small></td><td>${esc(r.region)}</td><td class="number">${num(r.km)} km</td><td class="number">${score(r.value)}${q.metric==='rising'&&r.historical_band===3?'<small>Already at highest harm level</small>':''}</td><td><button class="quiet" data-road="${esc(r.id)}" aria-label="View ${esc(r.id)}">↗</button></td></tr>`).join(''):'<tr><td colspan="5" class="empty">No roads match. Try a broader search.</td></tr>';
  $('page-label').textContent=rows.length?`${start+1}–${Math.min(start+pageSize,rows.length)} of ${integer(rows.length)} listed`:'No rows';
  $('previous-page').disabled=state.page===0;$('next-page').disabled=state.page>=totalPages-1;
}
function chartLegend(probability=false){return '<div class="chart-key"><span><i class="predicted-key"></i>'+(probability?'Predicted chance':'Predicted')+'</span><span><i class="actual-key"></i>'+(probability?'Actually occurred: Yes / No':'Actual')+'</span></div>';}
function chartTracks(row,history,metric){
  const risk=isRisk(metric);
  return '<span class="chart-tracks"><span class="chart-track"><i class="predicted-bar" style="width:'+row.predictedWidth+'%"></i></span>'+(history&&!risk?'<span class="chart-track"><i class="actual-bar" style="width:'+row.actualWidth+'%"></i></span>':'')+'</span><span class="chart-values"><b>'+score(row.predicted,metric)+'</b>'+(history?'<b class="actual-value'+(risk?' outcome-chip '+(row.actual?'occurred':'not-occurred'):'')+'">'+actualLabel(row.actual,metric)+'</b>':'')+'</span>';
}
function paintAccuracy(){
  const panel=$('accuracy-summary'),q=state.query;
  if(q.dataset!=='history'){panel.innerHTML='';panel.hidden=true;return;}
  panel.hidden=false;
  const e=state.result.summary.evaluation;
  if(!e.n){panel.innerHTML='<p class="muted">'+(e.excluded?'These roads were already at the highest harm level. No higher level can be checked.':'No matching roads to check.')+'</p>';return;}
  if(!isRisk(q.metric)){
    panel.innerHTML='<p class="accuracy-note">'+roadCount(e.n)+' checked · '+num(e.coverage_95*100)+'% of actual counts within the 95% prediction ranges.</p>';
    return;
  }
  const note=e.excluded?' · '+integer(e.excluded)+' already-high roads excluded':'';
  const bins=e.calibration.map(b=>'<div class="calibration-row"><span>'+integer(b.lower*100)+'–'+integer(b.upper*100)+'%<small>'+roadCount(b.n)+'</small></span><span class="chart-tracks"><span class="chart-track"><i class="predicted-bar" style="width:'+b.predicted*100+'%"></i></span><span class="chart-track"><i class="actual-bar" style="width:'+b.actual*100+'%"></i></span></span><span class="chart-values"><b>'+num(b.predicted*100)+'%</b><b class="actual-value">'+num(b.actual*100)+'%</b></span></div>').join('');
  panel.innerHTML='<h3>Predicted chance vs actual frequency</h3><p class="accuracy-note">'+roadCount(e.n)+' checked'+note+'</p><div class="chart-key"><span><i class="predicted-key"></i>Predicted chance</span><span><i class="actual-key"></i>Actual frequency</span></div><div class="calibration-chart">'+bins+'</div>';
}
function paintCharts(){
  if(!state.result)return;
  const q=state.query,history=q.dataset==='history',chart=RoadSafetyCharts.ranked(listRows(),q.metric,history,state.chartLimit);
  paintAccuracy();
  $('results-chart').innerHTML=(history?chartLegend(isRisk(q.metric)):'')+(chart.rows.length?`<p class="chart-caption">${integer(chart.rows.length)} roads shown · Select a bar to locate a road</p><div class="rank-chart">${chart.rows.map((r,i)=>`<button class="chart-row" data-chart-road="${esc(r.id)}" aria-pressed="${r.id===state.selected}" aria-label="${esc(r.road)}, ${esc(r.region)}, ${esc(r.id)}: ${history?'predicted ':''}${score(r.predicted,q.metric)}${history?', actual '+actualLabel(r.actual,q.metric):''}" title="${esc(r.id)}"><span class="chart-road"><b><small>${i+1}</small>${esc(r.road)}</b><span>${esc(r.region)}</span></span>${chartTracks(r,history,q.metric)}</button>`).join('')}</div>`:'<p class="empty">No roads match these filters.</p>');
}
function yearChart(data,metric,history){
  const chart=RoadSafetyCharts.years(data.periods,metric,history);
  return (history?chartLegend(isRisk(metric)):'')+'<div class="year-chart">'+chart.rows.map(r=>`<div class="year-chart-row"><span>${r.road}</span>${chartTracks(r,history,chart.metric)}</div>`).join('')+'</div>';
}
async function selectRoad(id,zoom=true){
  if(!state.result)return;
  RoadEngineering.clear();
  const request=state.request,selectionRequest=++state.selectionRequest,q={...state.query,id},data=await api('/api/road?'+new URLSearchParams(q));
  if(request!==state.request||selectionRequest!==state.selectionRequest)return;
  if(state.selected!==id)state.clickPoint=null;
  state.selected=id;state.selectedData=data;const r=data.road,v=data.values,future=q.dataset==='future';
  const range=(a,b)=>`${integer(a)}–${integer(b)}`;
  let content=`<p class="eyebrow">SELECTED ROAD</p><h2>${esc(r.road)}</h2><p class="detail-meta">${esc(r.region)}<br>${num(r.km)} km · ${periodText(q.period)}</p><div class="detail-location"><button class="secondary" id="street-selected" ${data.geometry_available?'':'disabled'}>Zoom to street level ↗</button></div>`;
  if(future){
    content+=`<div class="detail-metric"><span>Expected deaths</span><strong>${num(v[0],2)}</strong><small>95% range: ${range(v[2],v[3])} people</small></div><div class="detail-metric"><span>Expected serious injuries</span><strong>${num(v[1],2)}</strong><small>95% range: ${range(v[4],v[5])} people</small></div>`;
    if(['high','rising'].includes(q.metric))content+=`<div class="detail-foot">Chance of high harm: <strong>${num(v[6]*100)}%</strong><br>Chance of worsening harm: <strong>${num(v[7]*100)}%</strong>${r.historical_band===3?'<br><span>Already at the highest past harm level</span>':''}</div>`;
    content+='<h3 class="chart-heading">'+outcomeNames[q.metric]+' by year</h3>'+yearChart(data,q.metric,false);
  }else{
    const risk=isRisk(q.metric),k={fatal:0,serious:1,high:6,rising:7}[q.metric],actual=v[observedIndex[q.metric]],origin=data.period_metadata[q.period].origin;
    content+='<div class="detail-metric"><span>'+esc(labels[q.metric])+'</span><strong>'+score(v[k],q.metric)+'</strong>'+(risk?'':'<small>95% range: '+range(v[k===0?2:4],v[k===0?3:5])+' people</small>')+'</div><div class="detail-metric"><span>'+(risk?'Actually occurred':'Actual '+outcomeNames[q.metric].toLowerCase())+'</span><strong>'+actualLabel(actual,q.metric)+'</strong><small>Recorded in '+q.period+'</small></div>';
    if(q.metric==='rising')content+='<p class="detail-foot">Compared with '+(origin-1)+'–'+origin+'.'+(v[12]===3?' Already at the highest past harm level.':'')+'</p>';
    content+='<h3 class="chart-heading">'+outcomeNames[q.metric]+' by year</h3>'+yearChart(data,q.metric,true);
    content+='<p class="fine">For '+q.period+', history ends in '+origin+'.</p>';
  }
  if(!data.geometry_available)content+='<p class="band-warning">No matched road geometry. This unit remains in all numerical results.</p>';
  content+=`<button class="secondary" id="focus-selected" ${data.geometry_available?'':'disabled'}>Show full road extent ↗</button>`;
  content+=`<details class="road-reference"><summary>Road reference</summary><p>${esc(r.id)}</p></details>`;
  $('detail').innerHTML=content;bind('focus-selected',()=>focusRoad(id));bind('street-selected',()=>streetLevel(id));paintList();document.querySelectorAll('[data-chart-road]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.chartRoad===id)));highlight();paintMapSelection();if(zoom)streetLevel(id);
  RoadEngineering.select(r,q);
}

function initMap(){
  try{
    const map=new maplibregl.Map({container:'map',center:[-1.9,53],zoom:5.25,minZoom:3,maxZoom:19,attributionControl:false,style:{version:8,sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#e3edf3'}}]}});
    state.map=map;map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');map.addControl(new maplibregl.ScaleControl({unit:'metric',maxWidth:100}),'bottom-left');map.addControl(new maplibregl.AttributionControl({compact:false,customAttribution:'OS data © Crown copyright 2026 · Natural Earth'}),'bottom-right');
    map.on('load',async()=>{try{
      const outline=await api('/map/outline');map.addSource('land',{type:'geojson',data:outline});map.addLayer({id:'land',type:'fill',source:'land',paint:{'fill-color':'#f5f8fa','fill-outline-color':'#b5c9d5'}});
      map.addSource('roads',{type:'geojson',data:{type:'FeatureCollection',features:[]},promoteId:'id'});
      map.addLayer({id:'road-halo',type:'line',source:'roads',paint:{'line-color':'#fff','line-width':['interpolate',['linear'],['zoom'],4,2,7,4,15,7],'line-opacity':.8}});
      map.addLayer({id:'road-lines',type:'line',source:'roads',paint:{'line-color':['get','colour'],'line-width':['interpolate',['linear'],['zoom'],4,1.2,7,2.2,15,4.5],'line-opacity':.82}});
      map.addLayer({id:'selected-road',type:'line',source:'roads',filter:['==',['get','id'],''],paint:{'line-color':'#172f43','line-width':8,'line-opacity':.8}});
      map.addLayer({id:'selected-centre',type:'line',source:'roads',filter:['==',['get','id'],''],paint:{'line-color':'#f3d34b','line-width':3}});
      map.addLayer({id:'road-hit',type:'line',source:'roads',paint:{'line-color':'#fff','line-width':18,'line-opacity':0}});
      state.cityMarkers=[];
      for(const [name,lon,lat] of [['London',-.1276,51.5072],['Birmingham',-1.89,52.4862],['Manchester',-2.2426,53.4808],['Leeds',-1.5491,53.8008],['Newcastle',-1.6178,54.9783],['Bristol',-2.5879,51.4545],['Edinburgh',-3.1883,55.9533],['Cardiff',-3.1791,51.4816]]){
        const el=document.createElement('span');el.className='city-label';el.textContent=name;el.style.cssText='font:12px Arial;color:#4b6475;background:#ffffffcc;padding:2px 4px;border-radius:3px;pointer-events:none';state.cityMarkers.push(new maplibregl.Marker({element:el,anchor:'bottom-left'}).setLngLat([lon,lat]).addTo(map));
      }
      state.mapReady=true;setStreetMap();if(state.result)await updateMap();
    }catch(err){mapFallback(err.message);}});
    map.on('click','road-hit',e=>{const id=e.features?.[0]?.properties.id;if(id)selectRoad(id,false).then(()=>{if(state.selected===id){state.clickPoint=[e.lngLat.lng,e.lngLat.lat];refreshRoadNames();}}).catch(err=>error(err.message));});
    const tooltip=new maplibregl.Popup({closeButton:false,closeOnClick:false,offset:12});state.tooltip=tooltip;
    map.on('mousemove','road-hit',e=>{map.getCanvas().style.cursor='pointer';const f=e.features?.[0];if(f)tooltip.setLngLat(e.lngLat).setHTML(`<div class="road-tooltip"><strong>${esc(f.properties.road||f.properties.id)}</strong><br>${esc(labels[state.query?.metric]||'Road')}: ${score(f.properties.value)}<br>Click for road details</div>`).addTo(map);});
    map.on('mouseleave','road-hit',()=>{map.getCanvas().style.cursor='';tooltip.remove();});
    map.on('movestart',()=>tooltip.remove());map.on('moveend',refreshRoadNames);
    map.on('sourcedata',e=>{if(e.sourceId==='osm'&&e.sourceDataType==='content'&&e.tile){state.streetLoaded=true;if($('street-map').checked)$('map-chip').textContent='STREET MAP · NAMES & PLACES';}});
    map.on('error',e=>{const tileError=e.sourceId==='osm'||String(e.error?.message||'').includes('tile.openstreetmap.org');if(tileError&&$('street-map').checked){state.streetErrors++;if(state.streetErrors>=3&&!state.streetLoaded){$('street-map').checked=false;setStreetMap();$('map-chip').textContent='OFFLINE ROAD MAP · STREET SERVICE UNAVAILABLE';toast('Street maps could not connect. Road names and local forecasts are still available. You can retry the street-map switch.');}}else console.warn('Map rendering:',e.error?.message);});
  }catch(err){mapFallback(err.message);}
}
function mapFallback(message){$('map-fallback').hidden=false;$('map-chip').textContent='ROAD LIST AVAILABLE';console.warn(message);}
async function updateMap(){
  if(!state.mapReady||!state.result)return;
  const request=state.request,name=state.query.dataset;
  if(!state.geo[name])state.geo[name]=await api('/map/'+name);
  if(request!==state.request)return;
  const values=new Map(state.result.map),sorted=[...values.values()].sort((a,b)=>a-b),percent=['high','rising'].includes(state.query.metric);
  const cuts=percent?[.2,.4,.6,.8]:[.2,.4,.6,.8].map(q=>sorted[Math.min(sorted.length-1,Math.floor(sorted.length*q))]??0);
  const roadLabels=state.meta.datasets[name].road_labels||{};
  state.currentGeo={type:'FeatureCollection',features:state.geo[name].features.filter(f=>values.has(f.properties.id)).map(f=>{const value=values.get(f.properties.id),idx=cuts.filter(c=>value>c).length;return {...f,properties:{...f.properties,road:roadLabels[f.properties.id]||f.properties.id,value,colour:colours[idx]}};})};
  state.map.getSource('roads').setData(state.currentGeo);highlight();
  for(const layer of ['road-halo','road-lines','road-hit','selected-road','selected-centre'])state.map.setLayoutProperty(layer,'visibility','visible');
  state.map.resize();
  $('legend').innerHTML=sorted.length?'<span class="legend-title">'+(percent?'Chance (%)':'Expected people')+'</span>'+colours.map((c,i)=>{if(!percent&&((i>0&&i<4&&cuts[i]===cuts[i-1])||(i===4&&sorted[sorted.length-1]<=cuts[3])))return '';return `<span class="legend-item"><i class="swatch" style="background:${c}"></i>${i===4?'>'+score(cuts[3]):'≤'+score(cuts[i])}</span>`;}).join(''):'No roads in this selection';
  const missing=state.result.summary.missing_map;
  $('map-note').textContent=missing?`${missing} road units have no mapped shape. They remain in results and downloads.`:'';
  fitRoads();
}
function coordinates(geometry){return geometry.type==='LineString'?geometry.coordinates:geometry.type==='MultiLineString'?geometry.coordinates.flat():[];}
function fitFeatures(features){
  if(!state.mapReady||!features?.length)return;
  const b=new maplibregl.LngLatBounds();for(const f of features)for(const c of coordinates(f.geometry))b.extend(c);
  if(!b.isEmpty())state.map.fitBounds(b,{padding:{top:50,bottom:45,left:40,right:50},maxZoom:14,duration:window.matchMedia('(prefers-reduced-motion: reduce)').matches?0:500});
}
function fitRoads(){if(state.currentGeo?.features.length)fitFeatures(state.currentGeo.features);else if(state.mapReady)state.map.easeTo({center:[-1.9,53],zoom:5.25,duration:0});}
function focusRoad(id){fitFeatures(state.currentGeo?.features.filter(f=>f.properties.id===id));}
function highlight(){if(!state.mapReady)return;for(const name of ['selected-road','selected-centre'])state.map.setFilter(name,['==',['get','id'],state.selected||'']);refreshRoadNames();}
function setStreetMap(){
  if(!state.mapReady)return;
  const map=state.map,enabled=$('street-map').checked;
  if(enabled){state.streetErrors=0;state.streetLoaded=false;}
  if(enabled&&!map.getSource('osm')){
    map.addSource('osm',{type:'raster',tiles:['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],tileSize:256,maxzoom:19,attribution:'© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap contributors</a>'});
    map.addLayer({id:'osm',type:'raster',source:'osm',paint:{'raster-opacity':1}},'road-halo');
  }
  if(map.getLayer('osm'))map.setLayoutProperty('osm','visibility',enabled?'visible':'none');
  for(const marker of state.cityMarkers||[])marker.getElement().hidden=enabled;
  $('map-chip').textContent=enabled?'STREET MAP · ONLINE':'OFFLINE ROAD MAP · ROUTE LABELS';
  refreshRoadNames();
}

function roadPoint(id){
  if(id===state.selected&&state.clickPoint)return state.clickPoint;
  const feature=state.currentGeo?.features.find(f=>f.properties.id===id);if(!feature)return null;
  const lines=feature.geometry.type==='LineString'?[feature.geometry.coordinates]:feature.geometry.coordinates;
  let longest=null,best=0;
  for(const line of lines){let total=0;for(let i=1;i<line.length;i++)total+=Math.hypot((line[i][0]-line[i-1][0])*Math.cos(line[i][1]*Math.PI/180),line[i][1]-line[i-1][1]);if(total>best){best=total;longest=line;}}
  if(!longest)return lines[0]?.[0]||null;
  let distance=0;for(let i=1;i<longest.length;i++){const a=longest[i-1],b=longest[i],segment=Math.hypot((b[0]-a[0])*Math.cos(b[1]*Math.PI/180),b[1]-a[1]);if(distance+segment>=best/2){const t=segment?(best/2-distance)/segment:0;return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];}distance+=segment;}return longest[0];
}
function streetLevel(id){
  if(!state.mapReady){toast('The map is still loading. Try again in a moment.');return;}
  const point=roadPoint(id);if(!point){toast('No map geometry is available for this road unit.');return;}
  state.map.easeTo({center:point,zoom:16,duration:window.matchMedia('(prefers-reduced-motion: reduce)').matches?0:650});
  $('map').scrollIntoView({block:'nearest',behavior:'auto'});
}
function refreshRoadNames(){
  if(!state.mapReady)return;
  for(const marker of state.markers)marker.remove();state.markers=[];
  if(!state.result||!state.currentGeo?.features.length)return;
  const map=state.map,bounds=map.getBounds(),occupied=[],seen=new Set(),selected=state.currentGeo?.features.find(f=>f.properties.id===state.selected);
  const candidates=selected?[selected]:[];
  if(map.getZoom()>=9)candidates.push(...map.queryRenderedFeatures({layers:['road-lines']}));
  for(const f of candidates){const id=f.properties.id;if(seen.has(id))continue;seen.add(id);const isSelected=id===state.selected;let point=isSelected?roadPoint(id):coordinates(f.geometry).find(p=>bounds.contains(p));if(!point||!bounds.contains(point))continue;const p=map.project(point);if(occupied.some(v=>Math.abs(v.x-p.x)<120&&Math.abs(v.y-p.y)<44))continue;occupied.push(p);const el=document.createElement('span');el.className='road-name-marker'+(isSelected?' is-selected':'');el.textContent=f.properties.road||id;state.markers.push(new maplibregl.Marker({element:el,anchor:'bottom',offset:[0,-7]}).setLngLat(point).addTo(map));if(state.markers.length>=14)break;}
}
function paintMapSelection(){
  const data=state.selectedData;
  if(!data){$('map-selection').innerHTML='<span>Select a road</span>';return;}
  const r=data.road;
  $('map-selection').innerHTML=`<div><strong>${esc(r.road)} <small>· ${esc(r.region)}</small></strong></div><div class="selection-actions"><button class="primary" id="map-street" ${data.geometry_available?'':'disabled'}>Street-level view ↗</button><button class="quiet" id="map-extent" ${data.geometry_available?'':'disabled'}>Full road extent</button></div>`;
  bind('map-street',()=>streetLevel(r.id));bind('map-extent',()=>focusRoad(r.id));
}

async function showEvidence(){
  const records=await api('/api/history/yearly'),rows=[];
  for(const record of records){
    const m=record.metrics;
    const counts=key=>num(m[key].predicted_total)+' / '+integer(m[key].observed_total);
    const rates=key=>m[key].n?num(m[key].predicted_mean*100)+'% / '+num(m[key].observed_mean*100)+'%':'—';
    rows.push('<tr><td>'+record.year+'</td><td class="number">'+counts('fatal')+'</td><td class="number">'+counts('serious')+'</td><td class="number">'+rates('high')+'</td><td class="number">'+rates('rising')+'</td></tr>');
  }
  $('evidence-table').innerHTML=rows.join('');$('evidence-dialog').showModal();
}

async function boot(){
  try{
    bind('data-button',()=>$('data-dialog').showModal());
    document.querySelectorAll('.view-tabs [data-mode]').forEach(b=>{b.addEventListener('animationend',()=>b.classList.remove('is-clicked'));b.addEventListener('click',()=>{b.classList.remove('is-clicked');void b.offsetWidth;b.classList.add('is-clicked');configure(b.dataset.mode).catch(err=>error(err.message));});});
    bind('filters',e=>{e.preventDefault();return run();},'submit');
    $('filters').addEventListener('change',clearResults);
    bind('reset-map',fitRoads);bind('street-map',setStreetMap,'change');
    bind('quick-tour',()=>configure('future',{region:'West Midlands',period:'2027',metric:'serious',search:'A4040'}));
    bind('previous-page',()=>{state.page--;paintList();});bind('next-page',()=>{state.page++;paintList();});
    bind('road-table',e=>{const button=e.target.closest('[data-road]');if(button)return selectRoad(button.dataset.road);});
    bind('results-chart',e=>{const button=e.target.closest('[data-chart-road]');if(button)return selectRoad(button.dataset.chartRoad);});
    bind('chart-limit',()=>{state.chartLimit=Number($('chart-limit').value);paintCharts();},'change');
    bind('about-button',()=>$('about-dialog').showModal());bind('evidence-button',()=>{$('about-dialog').close();return showEvidence();});bind('go-history',()=>{$('evidence-dialog').close();return configure('history');});
    document.querySelectorAll('.dialog-close').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));
    $('data-dialog').addEventListener('close',()=>{if(location.hash==='#data-guide')history.replaceState(null,'',location.pathname+location.search);});
    state.meta=await api('/api/meta');$('bundle-label').textContent='Verified local bundle: '+state.meta.bundle_id;initMap();
    document.querySelectorAll('.view-tabs [data-mode], #quick-tour').forEach(button=>button.disabled=false);
    clearResults();
    if(location.hash==='#data-guide')$('data-dialog').showModal();
    window.addEventListener('hashchange',()=>{if(location.hash==='#data-guide'&&!$('data-dialog').open)$('data-dialog').showModal();});
  }catch(err){error('Cannot load the configured project. '+err.message);}
}
boot();
