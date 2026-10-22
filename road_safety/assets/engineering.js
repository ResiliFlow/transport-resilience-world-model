'use strict';
window.RoadEngineering = (() => {
  let ticket=0, selected=null, scenarioResult=null, context=null, checkTicket=0, scenarioTicket=0;
  const el=id=>document.getElementById(id);
  const safe=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n=(v,d=2)=>v===null?'—':Number(v).toLocaleString('en-GB',{minimumFractionDigits:d,maximumFractionDigits:d});
  async function request(path,body){
    const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
    const data=await r.json();if(!r.ok)throw Error(data.error||'Engineering results could not load.');return data;
  }
  function clear(){
    ticket++;checkTicket++;scenarioTicket++;selected=null;scenarioResult=null;context=null;
    el('engineering-card').hidden=true;el('engineering-body').innerHTML='';
    el('engineering-jump').disabled=true;
    el('engineering-jump').textContent='SPF & CMF · Select a road to explore';
    if(el('spf-check-dialog').open)el('spf-check-dialog').close();
  }
  function bars(rows){
    const max=Math.max(1,...rows.filter(r=>r.n!==0).flatMap(r=>[r.predicted,r.actual]));
    return '<div class="eng-key"><span><i></i>SPF estimate</span><span><i class="observed"></i>Recorded crashes</span></div><div class="eng-years">'+rows.map(r=>r.n===0?'<div class="eng-year eng-no-data"><b>'+r.year+'</b><span>No complete traffic data</span><span>—</span></div>':'<div class="eng-year"><b>'+r.year+'</b><div><span><i style="width:'+100*r.predicted/max+'%"></i></span><span><i class="observed" style="width:'+100*r.actual/max+'%"></i></span></div><div class="eng-year-values"><span>'+n(r.predicted,1)+'</span><span>'+n(r.actual,0)+'</span></div></div>').join('')+'</div>';
  }
  async function select(road,query){
    clear();const mine=ticket;context={...query};
    el('engineering-card').hidden=false;el('engineering-body').innerHTML='<p class="eng-loading" role="status">Loading SPF &amp; CMF…</p>';
    el('engineering-road').textContent=road.road+' · '+road.id;
    try{
      const data=await request('/api/engineering/road?'+new URLSearchParams({id:road.id}));
      if(mine!==ticket)return;
      selected=data;el('engineering-card').dataset.road=road.id;
      el('engineering-jump').disabled=false;el('engineering-jump').textContent='SPF & CMF · '+road.road+' →';
      const c=data.cmf;
      const baseline=data.available?'<div class="eng-value"><strong id="spf-value">'+n(data.expected)+'</strong><span>injury crashes / year</span></div>':'<p class="eng-notice">'+safe(data.unavailable_reason)+'</p>';
      el('engineering-body').innerHTML='<div class="eng-grid"><section class="eng-pane"><div class="eng-label"><b>SPF</b><h3>Road crash baseline</h3></div><p class="eng-full-name">Safety Performance Function</p>'+baseline+'<p class="eng-reference">'+data.baseline_year+' reference conditions · '+n(data.length_km,1)+' km</p>'+(data.baseline_year<2025?'<p class="eng-age">Latest complete traffic year: '+data.baseline_year+'. Updated traffic is needed for a current-year estimate.</p>':'')+'<dl class="eng-inputs"><div><dt>Daily traffic</dt><dd>'+n(data.aadf,0)+' vehicles</dd></div><div><dt>Road type</dt><dd>'+(data.motorway?'Motorway':data.a_motorway?'A(M) road':'A-road')+'</dd></div></dl><button class="secondary" id="spf-check-button">Check SPF against recorded crashes ↗</button><details class="eng-details"><summary>How is SPF calculated?</summary><p>Annual injury crashes estimated from road length, daily traffic, road class and year. A negative-binomial model is fitted to the existing network. This is a road-class reference, not a site-calibrated untreated baseline or the 2027–2030 injury forecast.</p><button class="text-link" id="spf-source-button">Model, data &amp; validation →</button></details></section>'+
      '<section class="eng-pane eng-cmf"><div class="eng-label"><b>CMF</b><h3>Compare a safety measure</h3></div><p class="eng-full-name">Crash Modification Factor</p><label class="eng-measure-label" for="cmf-measure">Measure</label><select id="cmf-measure"><option value="">Choose a measure…</option><option value="'+safe(c.id)+'">Permanent average-speed cameras</option></select><div id="cmf-content" hidden><div class="cmf-number"><strong>'+n(c.factor,3)+'</strong><span>CMF · injury crashes<br><small>Published study estimate</small></span></div><p class="eng-study">'+n((1-c.factor)*100,0)+'% estimated reduction in the source study</p><details class="eng-details"><summary>Source evidence</summary><p>'+safe(c.citation)+'. '+safe(c.location)+'. '+safe(c.scope)+'</p><p>Study 95% interval: '+n(c.lower,3)+'–'+n(c.upper,3)+'. '+safe(c.method)+'</p><p>This release provides whole-unit reference scenarios for non-motorway A-roads. Partial-road or combined-scheme estimates need separate inputs and evidence.</p><a class="text-link" href="'+safe(c.url)+'" target="_blank" rel="noopener">Read the original report ↗</a></details>'+(data.cmf_supported?'':'<p class="eng-notice" id="cmf-unavailable">'+safe(data.cmf_reason)+'</p>')+'<div id="cmf-output" aria-live="polite" aria-atomic="true"></div></div></section></div>';
      el('spf-check-button').addEventListener('click',()=>showChecks().catch(showError));
      el('spf-source-button').addEventListener('click',()=>showModel().catch(showError));
      el('cmf-measure').addEventListener('change',()=>{resetScenario();el('cmf-content').hidden=!el('cmf-measure').value;if(el('cmf-measure').value&&data.cmf_supported)calculate(mine);});
    }catch(err){if(mine===ticket){el('engineering-body').innerHTML='<p class="eng-notice" role="alert">'+safe(err.message)+'</p>';}}
  }
  function resetScenario(){
    scenarioTicket++;scenarioResult=null;el('cmf-output').innerHTML='';
    el('cmf-output').removeAttribute('aria-busy');
  }
  function showError(err){if(el('cmf-output'))el('cmf-output').innerHTML='<p class="eng-notice" role="alert">'+safe(err.message)+'</p>';}
  async function calculate(mine){
    const attempt=++scenarioTicket;
    const settings={id:selected.id,measure:el('cmf-measure').value,scenario_type:'fixed_reference'};
    scenarioResult=null;el('cmf-output').innerHTML='<p class="eng-loading">Calculating reference estimate…</p>';el('cmf-output').setAttribute('aria-busy','true');
    try{
      const data=await request('/api/engineering/scenario',settings);
      if(mine!==ticket||attempt!==scenarioTicket)return;
      if(data.status!=='estimated'){
        el('cmf-output').innerHTML='<div class="eng-notice eng-feedback"><h4>Reference estimate unavailable</h4><ul>'+data.issues.map(i=>'<li>'+safe(i.message)+'</li>').join('')+'</ul></div>';
        return;
      }
      scenarioResult=data;
      el('cmf-output').innerHTML='<div class="eng-scenario"><h4>Reference estimate</h4><p>'+data.baseline_year+' reference traffic · injury crashes / year</p><div class="eng-scenario-row"><span>Without new measure</span><strong>'+n(data.baseline)+'</strong><i style="width:100%"></i></div><div class="eng-scenario-row after"><span>With new measure</span><strong>'+n(data.after)+'</strong><i style="width:'+data.cmf*100+'%"></i></div><p class="eng-saving"><b>'+n(data.reduction)+' fewer / year</b> · '+n(data.reduction_percent,0)+'%</p><details class="eng-details"><summary>Estimate range</summary><p>CMF-only range: '+n(data.cmf_only_after_range[0])+'–'+n(data.cmf_only_after_range[1])+' crashes / year. This includes source-factor uncertainty only, not a local prediction interval or a verified site benefit.</p></details></div><section class="eng-conditions" aria-labelledby="cmf-conditions-title"><h4 id="cmf-conditions-title">Conditions for this estimate</h4><p class="eng-conditions-note">'+safe(data.conditions_note)+'</p><ul>'+data.conditions_for_estimate.map(text=>'<li>'+safe(text)+'</li>').join('')+'</ul><p class="eng-traffic-note">'+safe(data.comparison_assumptions.traffic)+'</p></section><button class="secondary eng-download" id="cmf-download">Download reference estimate</button>';
      el('cmf-download').addEventListener('click',download);
    }catch(err){if(mine===ticket&&attempt===scenarioTicket){showError(err);el('cmf-output').insertAdjacentHTML('beforeend','<button class="secondary" id="cmf-retry">Retry reference estimate</button>');el('cmf-retry').addEventListener('click',()=>calculate(mine));}}finally{if(mine===ticket&&attempt===scenarioTicket&&el('cmf-output'))el('cmf-output').removeAttribute('aria-busy');}
  }
  function download(){
    if(!scenarioResult)return;
    const r=scenarioResult;
    const rows=[['road_unit','reference_year','outcome','spf_baseline_per_year','cmf','with_measure_per_year','reduction_per_year','cmf_lower_95','cmf_upper_95','measure','scenario_type','source','source_url','scope'],[r.id,r.baseline_year,r.unit,r.baseline,r.cmf,r.after,r.reduction,...r.cmf_95,'Permanent average-speed cameras',r.scenario_type,r.source,r.source_url,r.scope],[],['condition_for_estimate','verification_status'],...r.conditions_for_estimate.map(text=>[text,'Assumed, not verified on site']),[],['modelling_assumption','description'],...Object.entries(r.comparison_assumptions),[],['site_conditions_verified',r.site_conditions_verified],['uncertainty',r.range_note]];
    const csv='\ufeff'+rows.map(row=>row.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='SPF_CMF_reference_'+r.id+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function showChecks(){
    const mine=++checkTicket,road=selected,ctx={...context};
    if(!road)return;
    const all=await request('/api/engineering/checks?'+new URLSearchParams({region:ctx.region,search:ctx.search}));
    if(mine!==checkTicket||selected!==road)return;
    el('spf-check-title').textContent='SPF prediction checks';
    const s=road.check_summary;
    el('spf-check-body').innerHTML='<p class="eng-dialog-intro">Model fitted through 2019. Each check uses the recorded traffic in 2020–2025, not that year’s crashes as inputs.</p><h3>'+safe(road.id)+' · injury crashes</h3>'+(s.n?bars(road.history)+'<p class="eng-check-note">Average error: <b>'+n(s.mae)+'</b> crashes / year · '+s.n+' years with complete traffic.</p>':'<p>No held-out years have complete traffic for this road.</p>')+'<h3>All roads matching your filters</h3><p class="eng-check-note">'+n(all.summary.n,0)+' road-year checks · average error <b>'+n(all.summary.mae)+'</b> crashes per road/year</p>'+bars(all.years)+'<p class="eng-check-note">Years with incomplete traffic are excluded. SPF checks are separate from the injury-forecast checks and do not validate CMF effects.</p>';
    el('spf-check-dialog').showModal();
  }
  async function showModel(){
    const mine=++checkTicket;
    const d=await request('/api/engineering/meta');if(mine!==checkTicket)return;
    const m=d.model;
    el('spf-check-title').textContent='SPF model & data';
    el('spf-check-body').innerHTML='<p>Safety Performance Function for annual police-recorded injury crashes. A crash is an event, not a person.</p><h3>Data and coverage</h3><p>'+n(m.network_units,0)+' existing road units. '+n(m.eligible_road_years,0)+' complete road-year records from 2005–2025; '+n(m.excluded_road_years,0)+' incomplete records excluded. Traffic coverage must reach 99%. Each baseline uses the latest complete reference year shown on the card.</p><p>Data sources and reuse permissions must be documented with the supplied bundle.</p><h3>Model specification</h3><p class="eng-formula">Expected crashes = length (km) × exp(b₀ + b₁ ln(daily traffic / 10,000) + b₂ motorway + b₃ A(M) + b₄(year − 2015))</p><dl class="eng-inputs">'+Object.entries(d.coefficients).map(([k,v])=>'<div><dt>'+safe(k.replaceAll('_',' '))+'</dt><dd>'+n(v,6)+'</dd></div>').join('')+'</dl><p>Negative-binomial dispersion: '+n(m.current_model.dispersion,6)+'. The current fit converged. Site effects and treatment history are not model inputs. The reference is not an official or site-calibrated untreated SPF. No site confidence interval is claimed.</p><h3>Held-out checks</h3><p>Fixed 2005–2019 fit, evaluated on '+n(m.test_summary.n,0)+' eligible 2020–2025 road-years with recorded target-year traffic. SPF mean absolute error: '+n(m.test_summary.mae)+' crashes per road/year. A pre-2020 pooled traffic-rate baseline gives '+n(m.simple_baseline_summary.mae)+'. These are traffic-conditional checks, not future-traffic forecasts.</p><p class="eng-check-note">Model: '+safe(d.model_id)+'</p>';
    el('spf-check-dialog').showModal();
  }
  document.addEventListener('DOMContentLoaded',()=>{
    el('engineering-jump').addEventListener('click',()=>el('engineering-card').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'}));
  });
  return {clear,select};
})();
