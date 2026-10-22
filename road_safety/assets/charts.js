'use strict';
// Shared, deterministic chart scaling. No change to the frozen forecast values.
const RoadSafetyCharts = (() => {
  function ranked(rows, metric, history, limit=10) {
    limit=Math.max(1,Math.min(100,Math.floor(Number(limit)||10)));
    const observed={fatal:8,serious:9,high:10,rising:11};
    const selected=rows.slice(0,limit).map(r=>({id:r.id,road:r.road,region:r.region,predicted:Number(r.value),actual:history?Number(r.values[observed[metric]]):null}));
    const maximum=['high','rising'].includes(metric)?1:Math.max(0,...selected.flatMap(r=>history?[r.predicted,r.actual]:[r.predicted]));
    return {maximum,rows:selected.map(r=>({...r,predictedWidth:maximum?r.predicted/maximum*100:0,actualWidth:history&&maximum?r.actual/maximum*100:0}))};
  }
  function years(periods, metric, history) {
    const index={fatal:0,serious:1,high:6,rising:7}[metric];
    const rows=Object.keys(periods).filter(p=>/^\d{4}$/.test(p)).sort().map(p=>({id:p,road:p,region:'',value:periods[p][index],values:periods[p]}));
    const displayMetric=metric;
    return {...ranked(rows,displayMetric,history,rows.length),metric:displayMetric};
  }
  return {ranked,years};
})();
if(typeof module!=='undefined'&&module.exports)module.exports=RoadSafetyCharts;
