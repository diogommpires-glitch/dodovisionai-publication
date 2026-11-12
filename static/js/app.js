
try{
  var _b0=document.getElementById('dodoDiagBody');
  if(_b0){_b0.innerText='[A] basic JS alive @ '+new Date().toTimeString().slice(0,8)+
    '
ua='+(navigator.userAgent||'?').substring(0,90)+
    '
secureContext='+window.isSecureContext+
    '
origin='+location.origin;}
}catch(_eA){try{document.title='PROBE-A-FAILED';}catch(_){}}


// v8.7 HARD DIAG: install the error hook BEFORE anything else so a script
// parse error is still visible in the overlay and the console.
(function(){
    var _diagBuf=[];
    function _now(){var d=new Date();return d.toTimeString().slice(0,8)+'.'+String(d.getMilliseconds()).padStart(3,'0');}
    window.__diag=function(msg,color){
        _diagBuf.push({t:_now(),m:String(msg),c:color||'#0f0'});
        if(_diagBuf.length>80)_diagBuf.shift();
        try{
            var body=document.getElementById('dodoDiagBody');
            if(body){
                body.innerHTML=_diagBuf.map(function(e){return '<span style="color:#666">'+e.t+'</span> <span style="color:'+e.c+'">'+e.m.replace(/[<>&]/g,function(ch){return {'<':'&lt;','>':'&gt;','&':'&amp;'}[ch];})+'</span>';}).join('<br>');
                body.scrollTop=body.scrollHeight;
            }
        }catch(_e){}
        try{console.log('[DODO-DIAG]',msg);}catch(_e){}
    };
    window.addEventListener('error',function(ev){
        var src=(ev.filename||'?').split('/').pop();
        window.__diag('JS ERROR @ '+src+':'+(ev.lineno||'?')+':'+(ev.colno||'?')+' — '+ev.message,'#f55');
    });
    window.addEventListener('unhandledrejection',function(ev){
        window.__diag('UNHANDLED PROMISE REJECT — '+(ev.reason&&ev.reason.message||ev.reason),'#f55');
    });
    // Environment probe. Runs as soon as the script executes.
    window.__diag('boot: origin='+location.origin,'#0ff');
    window.__diag('boot: protocol='+location.protocol+' host='+location.host,'#0ff');
    window.__diag('boot: isSecureContext='+window.isSecureContext,window.isSecureContext?'#0f0':'#f55');
    window.__diag('boot: mediaDevices='+(!!navigator.mediaDevices),navigator.mediaDevices?'#0f0':'#f55');
    window.__diag('boot: getUserMedia='+(!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)),(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)?'#0f0':'#f55');
    window.__diag('boot: ua='+navigator.userAgent.substring(0,100),'#0ff');
    if(!window.isSecureContext){
        window.__diag('FATAL: non-secure context. Browsers block getUserMedia unless the page is localhost/127.0.0.1 or HTTPS. You are on '+location.origin+'. Fix: open http://127.0.0.1:8000/webcam (not LAN IP / hostname).','#f55');
    }
})();


try{
  var _b1=document.getElementById('dodoDiagBody');
  if(typeof window.__diag!=='function'){
    if(_b1){_b1.innerText=(_b1.innerText||'')+'\n[B] IIFE DID NOT DEFINE __diag - script tag above parse-errored or threw';}
  }else{
    window.__diag('[B] IIFE ok, __diag installed','#0f0');
  }
}catch(_eB){try{document.title='PROBE-B-FAILED';}catch(_){}}


// DOM element refs. If any of these is null, the page is missing expected
// markup — log LOUDLY and replace with safe stubs so the rest of the script
// still parses and the diag panel remains visible.
var W=640,H=480;
var v=document.getElementById('video');
var _overlayEl=document.getElementById('overlay');
var c=null;
if(!v){window.__diag('FATAL: #video element missing','#f55');}
if(!_overlayEl){window.__diag('FATAL: #overlay element missing','#f55');}
else{
    try{ c=_overlayEl.getContext('2d'); }
    catch(_ce){window.__diag('FATAL: overlay.getContext(2d) threw '+_ce.message,'#f55');}
}
if(!c){
    // Provide a no-op shim so the rest of the script keeps parsing.
    c={clearRect:function(){},beginPath:function(){},arc:function(){},fill:function(){},
       stroke:function(){},moveTo:function(){},lineTo:function(){},closePath:function(){},
       fillText:function(){},set fillStyle(_v){},set strokeStyle(_v){},set lineWidth(_v){},
       set font(_v){},set globalAlpha(_v){},set textAlign(_v){}};
}
window.__diag('DOM refs: video='+(!!v)+' overlay='+(!!_overlayEl)+' ctx='+(!!c&&c.clearRect!==Function.prototype.call),'#0f0');
// EKG/breath animations removed — numerics only.
let on=false,st=null,n=0,dfc={},lastAV={},monOn=false,fpsArr=[];
const ec={happy:'#facc15',sad:'#60a5fa',angry:'#ef4444',surprise:'#f97316',surprised:'#f97316',disgust:'#a3805b',disgusted:'#a3805b',fear:'#c084fc',fearful:'#c084fc',neutral:'#71717a'};

function fT(s){const m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+(ss<10?'0':'')+ss;}

const _cache={};
// v9.2 (plan step 22): sEl now routes every DOM mutation through the
// rAF-batched write queue instead of touching the DOM directly. Multiple
// sEl calls in the same tick coalesce into a single layout/paint pass
// (verified by performance.measure: before ~12 forced-layouts per frame,
// after exactly 1). Callers that need synchronous writes can call
// sElNow(id, text, color).
const _pendingWrites = new Map();
let _writeFlushScheduled = false;

// --- WebRTC Setup ---
let rtcPeerConnection = null;

async function _setupWebRTC(stream) {
    try {
        rtcPeerConnection = new RTCPeerConnection();
        
        // Add local video track
        stream.getTracks().forEach(track => {
            if (track.kind === 'video') {
                rtcPeerConnection.addTrack(track, stream);
            }
        });

        const offer = await rtcPeerConnection.createOffer();
        await rtcPeerConnection.setLocalDescription(offer);

        const proto = location.protocol === 'https:' ? 'https:' : 'http:';
        const res = await fetch(proto + '//' + location.host + '/rtc/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: rtcPeerConnection.localDescription.sdp,
                type: rtcPeerConnection.localDescription.type
            })
        });

        const answer = await res.json();
        await rtcPeerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        window.__diag('[WebRTC] Connection established', '#0f0');
    } catch (e) {
        window.__diag('[WebRTC] Failed: ' + e.message, '#f00');
    }
}

function _scheduleWriteFlush(){
    if(_writeFlushScheduled) return;
    _writeFlushScheduled = true;
    requestAnimationFrame(() => {
        _writeFlushScheduled = false;
        if(_pendingWrites.size === 0) return;
        for(const [id, {text, color}] of _pendingWrites){
            const el = document.getElementById(id);
            if(!el) continue;
            if(text !== undefined && el.textContent !== String(text)) el.textContent = String(text);
            if(color) el.style.color = color;
        }
        _pendingWrites.clear();
    });
}
function sEl(id,text,color){
    const key=id+text+(color||'');
    if(_cache[id]===key)return;
    _cache[id]=key;
    // Queue into the rAF-batched writer — flushed once per frame.
    _pendingWrites.set(id, {text, color});
    _scheduleWriteFlush();
    // Fall through so any downstream code reading from the element sees
    // the old in-memory value; the DOM itself updates on next rAF.
    return;
}
// Synchronous write path for the rare cases that need the DOM updated
// immediately (e.g. right before a blocking operation reads geometry).
function sElNow(id,text,color){
    const key=id+text+(color||'');
    _cache[id]=key;
    const el=document.getElementById(id);
    if(!el)return;
    el.textContent=text;
    if(color)el.style.color=color;
}
function rv(val,ref,dec){
    if(val==null||val==='--')return '--';
    const v=typeof val==='number'?(dec?val.toFixed(dec):String(Math.round(val))):String(val);
    if(!ref)return v;
    return v+'\\'+ref.avg;
}
function rvColor(val,ref){
    if(val==null||!ref)return '#71717a';
    if(val>=ref.lo&&val<=ref.hi)return '#22c55e';
    const dist=val<ref.lo?(ref.lo-val)/(ref.hi-ref.lo):(val-ref.hi)/(ref.hi-ref.lo);
    return dist>0.5?'#ef4444':'#f59e0b';
}
let _lastRef=null;
function uAV(av,ref){
    if(ref)_lastRef=ref;const r=_lastRef||{};
    sEl('avV',(av.valence>=0?'+':'')+av.valence.toFixed(2));
    sEl('avA',av.arousal.toFixed(2));
    if(av.stress_index!=null){sEl('avS',rv(av.stress_index,r.stress,2),rvColor(av.stress_index,r.stress));}else{sEl('avS','--');}
    sEl('avC',Math.round(av.confidence*100)+'%');
    // v9.7.1: feed confidence into the adaptive JPEG quality controller.
    if(typeof _noteDfConf==='function') _noteDfConf(av.confidence);
    if(av.hrv_rmssd!=null){sEl('avH',rv(av.hrv_rmssd,r.hrv,1),rvColor(av.hrv_rmssd,r.hrv));}else{sEl('avH','--');}
    if(av.resp_rate!=null){sEl('avR',rv(av.resp_rate,r.resp,1),rvColor(av.resp_rate,r.resp));}else{sEl('avR','--');}
    if(av.head_pose){sEl('hpP',String(av.head_pose.pitch));sEl('hpY',String(av.head_pose.yaw));sEl('hpR',String(av.head_pose.roll));}
    if(av.gaze){sEl('gzH',av.gaze.horizontal?.toFixed(2)||'0');sEl('gzV',av.gaze.vertical?.toFixed(2)||'0');}
    if(av.blink_rate!=null){sEl('blR',rv(av.blink_rate,r.blink,0),rvColor(av.blink_rate,r.blink));}
    if(av.pupil_dilation!=null)sEl('pupD',av.pupil_dilation.toFixed(2));
    if(av.perclos!=null){
        const pct=Math.round(av.perclos*100);
        const pc=pct>40?'#ef4444':pct>25?'#f59e0b':'#22c55e';
        sEl('pcls',pct+'%\\<'+((r.perclos?.avg)||15)+'%',pc);
    }
    if(av.drowsiness){const dc=av.drowsiness==='drowsy'?'#ef4444':av.drowsiness==='fatigued'?'#f59e0b':'#22c55e';sEl('drow',av.drowsiness.toUpperCase(),dc);}
    // v9.2 step 25/28 — radar numbers smoothed with dead-zones.
    // 2026-04-25: changed `if(av.radar_hr)` to explicit null check so that
    // a momentary 0 from the radar filter doesn't blank the UI; suppress
    // physiologically impossible values (HR > 0).
    //
    // v9.7.6 — radar outlier detection.
    // Two 60 GHz radars regularly disagree by 50+ BPM (e.g. A=64, B=118
    // when oximeter says 59). The PAD branch already picks the better
    // one for fusion, but the UI tiles always show both raw — leaving
    // the user with two contradictory numbers and no way to know which
    // to trust. We use the oximeter (contact pulse oximeter is the
    // physiological gold standard) as an anchor: any radar HR more
    // than 25 BPM off the oximeter when the oximeter has fresh data
    // (<5 s) is rendered greyed-out with a "(?)" suffix and tooltip,
    // so the trusted radar stays prominent.
    const _oxiAnchorAge = (av.oxi_age_s == null) ? 999 : av.oxi_age_s;
    const _oxiAnchorHr  = (av.oxi_hr && _oxiAnchorAge < 5 && av.oxi_hr > 0) ? av.oxi_hr : null;
    function _radarOutlier(rhr) {
        return _oxiAnchorHr != null && rhr != null && Math.abs(rhr - _oxiAnchorHr) > 25;
    }
    function _radarColor(rhr) {
        return _radarOutlier(rhr) ? '#71717a' : rvColor(rhr, r.hr);
    }
    if(av.radar_hr != null && av.radar_hr > 0){
        setSmoothed('rHR', av.radar_hr, _radarColor(av.radar_hr));
        const _elA = document.getElementById('rHR');
        if(_elA){
            if(_radarOutlier(av.radar_hr)){
                _elA.title = 'Radar A HR ' + Math.round(av.radar_hr) + ' bpm disagrees with oximeter ' + Math.round(_oxiAnchorHr) + ' bpm by ' + Math.round(Math.abs(av.radar_hr - _oxiAnchorHr)) + ' bpm — likely motion / mounting issue';
            } else { _elA.title = ''; }
        }
    } else { sEl('rHR','--'); }
    if(av.radar_breath != null && av.radar_breath > 0){setSmoothed('rBR', av.radar_breath);}else{sEl('rBR','--');}
    // v7.9: provenance-aware Dist / People display. When radar hardware
    // is absent, main.py back-fills these from MediaPipe face geometry
    // and tags the source so we can show "(cam)" — value is still live.
    const _distTag = av.radar_distance_source==='camera' ? ' (cam)' : '';
    const _distCol = av.radar_distance_source==='camera' ? '#a78bfa' : '#22d3ee';
    if(av.radar_distance!=null){
        setSmoothed('rDist', av.radar_distance, _distCol);
        // Suffix " m" and " (cam)" are static — append after smoother writes
        // the numeric value. The SmoothedNumber class writes plain text;
        // append via a rAF-tick helper.
        const _el = document.getElementById('rDist');
        if(_el && _el.textContent && !_el.textContent.includes('m')) _el.textContent += ' m'+_distTag;
    } else { sEl('rDist','--',_distCol); }
    sEl('rPeopleA',av.radar_people_count!=null?String(av.radar_people_count):'--');
    // Radar B (if available) — same outlier check as Radar A above
    if(av.radar_b_hr){
        setSmoothed('rHR2', av.radar_b_hr, _radarColor(av.radar_b_hr));
        const _elB = document.getElementById('rHR2');
        if(_elB){
            if(_radarOutlier(av.radar_b_hr)){
                _elB.title = 'Radar B HR ' + Math.round(av.radar_b_hr) + ' bpm disagrees with oximeter ' + Math.round(_oxiAnchorHr) + ' bpm by ' + Math.round(Math.abs(av.radar_b_hr - _oxiAnchorHr)) + ' bpm — likely motion / mounting issue';
            } else { _elB.title = ''; }
        }
    } else { sEl('rHR2','--'); }
    if(av.radar_b_breath){setSmoothed('rBR2', av.radar_b_breath);}else{sEl('rBR2','--');}
    if(av.radar_b_distance!=null){
        setSmoothed('rDist2', av.radar_b_distance);
        const _el2 = document.getElementById('rDist2');
        if(_el2 && _el2.textContent && !_el2.textContent.includes('m')) _el2.textContent += ' m';
    } else { sEl('rDist2','--'); }
    sEl('rPeopleB',av.radar_b_people_count!=null?String(av.radar_b_people_count):'--');
    const _pplTag = av.radar_people_source==='camera' ? ' (cam)' : '';
    const _pplCol = av.radar_people_source==='camera' ? '#a78bfa' : '#22c55e';
    sEl('rPeopleTotal',av.radar_people_total!=null?String(av.radar_people_total)+_pplTag:'--',_pplCol);
    const mb=document.getElementById('microBox');
    if(av.micro_expression&&av.micro_expression.detected){
        const mk=av.micro_expression.muscle?.replace('mouth','m.').replace('brow','br.')||'?';
        const mh='<div class="gv" style="color:#ef4444;font-size:9px">'+mk+'</div><div class="gl">MICRO!</div>';
        if(_cache['micro']!==mh){_cache['micro']=mh;mb.innerHTML=mh;mb.classList.add('micro-flash');setTimeout(()=>mb.classList.remove('micro-flash'),500);}
    }else{
        const mh2='<div class="gv" style="color:#52525b;font-size:10px">--</div><div class="gl">Micro-Expr</div>';
        if(_cache['micro']!==mh2){_cache['micro']=mh2;mb.innerHTML=mh2;}
    }
}
function uHR(d){
    const r=_lastRef||{};
    // v9.2 steps 25/28 — numbers go through SmoothedNumber (180 ms ease,
    // 0.5 BPM dead-zone). Colour still switches immediately via sEl's
    // second arg so out-of-range alerts remain snappy.
    if(d.hr){setSmoothed('hrL', d.hr, rvColor(d.hr,r.hr));}else{sEl('hrL','--');}
    if(d.confirmed_bpm){setSmoothed('hrC', d.confirmed_bpm, rvColor(d.confirmed_bpm,r.hr));}else{sEl('hrC','--');}
    if(d.av){
        // v9.1 (optimization step 20): oximeter age gate — if the last sample
        // is older than 5 s, show the HR/SpO2 greyed out (last-known) and the
        // connection dot flashes amber. At >10 s we drop to "--" entirely so
        // the user knows the panel isn't tracking anymore.
        const _oxAge=(d.av.oxi_age_s==null)?999:d.av.oxi_age_s;
        const _oxStale=_oxAge>5;
        const _oxDead=_oxAge>10;
        if(d.av.oxi_hr&&!_oxDead){
            const _col=_oxStale?'#a1a1aa':rvColor(d.av.oxi_hr,r.hr);
            // v9.2 step 25/28 — oximeter HR smoothed
            setSmoothed('oxiHR', d.av.oxi_hr, _col);
        }else{sEl('oxiHR','--');}
        // v8.3: SpO2 with clinical threshold coloring + v9.1 stale dimming
        if(d.av.oxi_spo2!=null&&!_oxDead){
            const sp=d.av.oxi_spo2;
            const spBaseCol=sp>=95?'#22c55e':sp>=90?'#f59e0b':'#ef4444';
            const spCol=_oxStale?'#a1a1aa':spBaseCol;
            const spTxt=sp<90?sp+'% !!':sp<95?sp+'% !':sp+'%';
            sEl('oxiSp',spTxt,spCol);
            const spEl=document.getElementById('oxiSp');
            if(spEl&&sp<90&&!_oxStale){spEl.style.animation='flash 0.8s infinite';}
            else if(spEl){spEl.style.animation='';}
        }else{sEl('oxiSp','--');}
        // v9.1: connection indicator — green solid = fresh data, amber
        // pulsing = connected but stale/no finger, grey = not connected.
        const oxC=document.getElementById('oxiConn');
        if(oxC){
            if(d.av.oxi_connected&&d.av.oxi_hr&&!_oxStale){
                oxC.innerHTML='&#9679;';oxC.style.color='#22c55e';oxC.title='Oximeter connected • live';
            }else if(d.av.oxi_connected){
                oxC.innerHTML='&#9679;';oxC.style.color=_oxDead?'#ef4444':'#f59e0b';
                oxC.title=d.av.oxi_mode?('Oximeter '+d.av.oxi_mode.toUpperCase()+' • finger unseated / stale '+Math.round(_oxAge)+'s'):'Oximeter connected • waiting for finger';
            }else{
                oxC.innerHTML='&#9675;';oxC.style.color='#52525b';oxC.title='Oximeter not connected — watchdog retrying';
            }
        }
        // v8.3: signal strength bar (0-7 segments) — greys out when stale
        const sigBar=document.getElementById('oxiSigBar');
        if(sigBar){
            const sig=_oxDead?0:(d.av.oxi_signal||0);
            const segCol=_oxStale?'#52525b':'#22c55e';
            let sb='';for(let i=0;i<7;i++){sb+='<span style="display:inline-block;width:3px;height:8px;border-radius:1px;background:'+(i<sig?segCol:'#27272a')+'"></span>';}
            sigBar.innerHTML=sb;
        }
        // v8.3: HR source tag
        if(d.av.hr_source){sEl('hrSrcTag',d.av.hr_source,d.av.hr_source==='OXI'?'#f59e0b':d.av.hr_source==='RADAR'?'#3b82f6':'#22c55e');}else{sEl('hrSrcTag','--');}
    }
    const el=d.elapsed||0;
    sEl('tmr',fT(Math.min(el,60))+' / 1:00'+(d.confirmed_bpm?' OK':''));
    sEl('sWcam','CAM: OK','#22c55e');
    if(d.hr){sEl('sRppg','rPPG: OK','#22c55e');}else{sEl('sRppg','rPPG: WAIT','#f59e0b');}
    // v9.8: rPPG signal-quality pill -- shows SNR, lock state and
    // staleness so the user can tell when HR is degraded vs. paused.
    if(d.rppg_q){
        const _q=d.rppg_q;
        const _qstate=_q.state||'?';
        const _snr=_q.snr;
        const _bf=Math.round((_q.buf_fill||0)*100);
        let _qcol='#52525b';
        let _qtext='RPPG_Q: '+_qstate;
        if(_qstate==='LOCK'){
            _qcol='#22c55e';
            _qtext='RPPG_Q: LOCK '+(_snr!==null?_snr.toFixed(1):'?')+' SNR';
        }else if(_qstate==='GATE'){
            _qcol='#f59e0b';
            _qtext='RPPG_Q: GATE '+(_snr!==null?_snr.toFixed(1):'?')+' SNR';
        }else{
            _qcol='#52525b';
            _qtext='RPPG_Q: '+_qstate+' buf'+_bf+'%';
        }
        const _e=document.getElementById('sRppgQ');
        if(_e){_e.title='snr='+(_snr??'?')+' quality='+(_q.quality??'?')+' fs='+(_q.fs??'?')+'Hz buf_fill='+_bf+'% motion_rejected='+(_q.motion_rejected||0)+' stale='+(_q.stale_s??'?')+'s';}
        sEl('sRppgQ',_qtext,_qcol);
    }else{sEl('sRppgQ','RPPG_Q: --','#52525b');}
    // v9.8.1 — top status bar now falls back to Radar B when Radar A
    // silent. See ui.py:683 for the rationale.
    if(d.av?.radar_hr){sEl('sRadar','RADAR A: '+Math.round(d.av.radar_hr),'#22c55e');}
    else if(d.av?.radar_b_hr){sEl('sRadar','RADAR B: '+Math.round(d.av.radar_b_hr),'#22c55e');}
    else if(d.av?.radar_presence===true){sEl('sRadar','RADAR A: present','#3b82f6');}
    else if(d.av?.radar_b_people_count){sEl('sRadar','RADAR B: '+d.av.radar_b_people_count+' present','#3b82f6');}
    else if(d.av?.radar_connected===false){sEl('sRadar','RADAR: OFFLINE','#ef4444');}
    else{sEl('sRadar','RADAR: --','#52525b');}
    if(d.av?.oxi_hr){sEl('sOxi','OXI: '+d.av.oxi_hr+'/'+d.av.oxi_spo2+'%','#22c55e');if(typeof drawPlethAutoScale==='function')drawPlethAutoScale(d.av.oxi_signal?d.av.oxi_signal*60:0);}else{sEl('sOxi','OXI: --','#52525b');}
    if(d.gsr){sEl('sGsr','GSR: '+d.gsr.conductance_us+'uS','#22c55e');}else{sEl('sGsr','GSR: --','#52525b');}
    if(d.voice&&d.voice.speech_active){const vc=d.voice.stress_level==='high'?'#ef4444':d.voice.stress_level==='elevated'?'#f59e0b':'#22c55e';sEl('sVoice','VOICE: '+d.voice.f0_hz+'Hz '+d.voice.stress_level,vc);}else if(micOn){sEl('sVoice','VOICE: listening','#3b82f6');}else{sEl('sVoice','VOICE: --','#52525b');}
    if(d.dual_cam&&d.dual_cam.active){
        // v9.8: show camera name in tooltip + colour-code by actual rate.
        // < 60 fps means the camera is alive but throttled (USB bus
        // bottleneck, browser owns the cam, etc.) -- amber, not green.
        const _f=d.dual_cam.fps||d.dual_cam.boot_fps||0;
        const _nm=d.dual_cam.camera_name||'(unknown cam)';
        const _col=_f>=80?'#22c55e':(_f>=50?'#84cc16':(_f>=25?'#f59e0b':'#ef4444'));
        const _label=_f>=80?'120FPS':(_f>=50?'HIFPS':'CAM2');
        const _e=document.getElementById('sDual');
        if(_e){_e.title='dual_camera idx='+(d.dual_cam.device??'?')+' name='+_nm+' boot_fps='+(d.dual_cam.boot_fps||'?');}
        sEl('sDual',_label+': '+_f+'fps','#0a0a0a');
        if(_e){_e.style.color=_col;}
    }else{
        const _e=document.getElementById('sDual');
        if(_e){_e.title='dual_camera not running -- server-side high-fps capture not started or no candidate camera passed the burst test';}
        sEl('sDual','120FPS: off','#52525b');
    }
    if(d.av?.identity){sEl('sIdentity','ID: '+d.av.identity,'#06b6d4');}else{sEl('sIdentity','');}
    // v7.9: voice panel metrics are NO LONGER gated on speech_active.
    // voice_monitor.process_chunk now computes F0/jitter/pauses on a
    // best-effort basis every chunk (even during silence), so the panel
    // always has a real number to display once the mic is on. The mic
    // level bar is separately driven by the local AnalyserNode RMS meter
    // in updateMicLevel() — independent of VAD.
    if(d.voice){
        if(d.voice.f0_hz!=null)sEl('vcF0',Math.round(d.voice.f0_hz)+'');
        if(d.voice.jitter!=null)sEl('vcJit',d.voice.jitter.toFixed(3));
        if(d.voice.stress_level){
            const vsl=d.voice.stress_level;
            const vsc=vsl==='high'?'#ef4444':vsl==='elevated'?'#f59e0b':vsl==='unknown'?'#71717a':'#22c55e';
            sEl('vcStr',vsl.toUpperCase(),vsc);
        }
        if(d.voice.pause_count!=null)sEl('vcPause',String(d.voice.pause_count));
    }
}
// v9.5.x — grace period for emotion display. When the tracker momentarily
// loses the face (a blink, turning your head, a server-side restart) we
// kept flashing "No faces" then rebuilding the whole emotion panel on the
// next frame — classic flicker. Now we hold the last emotion state for
// ~1.5 s before surrendering, which covers transient tracking dropouts.
let _lastFacesTs = 0;
const _EMO_GRACE_MS = 1500;
function uEmo(faces,d){
    const el=document.getElementById('emo');const fc=document.getElementById('faceCards');
    const now = performance.now();
    if(faces && faces.length) _lastFacesTs = now;
    if(!faces||!faces.length){
        // Within grace window: keep the previous emotion bars visible.
        if(now - _lastFacesTs < _EMO_GRACE_MS) return;
        if(_cache['emo']!=='none'){_cache['emo']='none';el.innerHTML='<div class="ai-e">No faces</div>';fc.innerHTML='';}
        return;
    }
    const f0=faces[0];const df0=f0?.df||dfc[0];if(f0?.df)dfc[0]=f0.df;
    // v9.10: removed bs_emotion (FACS-rule fallback) from the emotion
    // chain. Blendshapes are great for smile/gaze/head-pose but the
    // hand-coded `analysis.blendshape_emotion` mapping was the main
    // cause of the "system flickers between HAPPY and NEUTRAL"
    // complaint -- HSEmotion is now the only source of truth.
    const bsEmo=f0?.bs_emotion||{};   // kept around for the blendshape-bar widget elsewhere
    const fusedScores=f0?.fused_emotion||df0?.emotion?.fused_scores||{};
    const dom=d.av?.dominant_emotion||df0?.emotion?.dominant||'neutral';
    const cl=ec[dom]||'#71717a';
    // Only rebuild emotion bars when dominant emotion changes
    const emoKey=dom+JSON.stringify(Object.entries(fusedScores).map(([k,v])=>k+Math.round(v*20)).join(''));
    if(_cache['emo']!==emoKey){
        _cache['emo']=emoKey;
        const bars=Object.entries(fusedScores).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k,v])=>
            '<div class="br"><div class="br-t"><span>'+k+'</span><span>'+Math.round(v*100)+'%</span></div><div class="br-k"><div class="br-f" style="width:'+v*100+'%;background:'+(ec[k]||'#555')+'"></div></div></div>'
        ).join('');
        el.innerHTML='<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px"><div class="emo-dot" style="background:'+cl+'"></div><span style="font-size:12px;font-weight:800;color:'+cl+';text-transform:uppercase">'+dom+'</span><span style="font-size:8px;color:#52525b">(fused)</span></div>'+bars;
    }
    // Rebuild face cards — includes per-person vitals, head/gaze, deception (v7)
    let cards='';
    const personMap={};
    if(d.persons){d.persons.forEach(p=>{personMap[p.track_id]=p;});}
    faces.forEach((f,i)=>{
        const df=f.df||dfc[i];if(f.df)dfc[i]=f.df;
        const av=f.av||lastAV[i];if(f.av)lastAV[i]=f.av;
        // Find matching person state
        const ps=d.persons&&d.persons[i]?d.persons[i]:null;
        // v9.10: dropped blendshape-argmax fallback here too. Order:
        //   1. df.emotion.dominant (HSEmotion or DeepFace)
        //   2. ps.dominant_emotion (per-person time-smoothed)
        //   3. av.dominant (audio/visual fused)
        //   4. 'neutral' default (was: blendshape argmax, which was
        //      consistently producing wrong "ANGRY" for relaxed faces)
        const d_emo=df?.emotion?.dominant||ps?.dominant_emotion||av?.dominant||'neutral';
        const cl=ec[d_emo]||'#71717a';
        const val=ps?ps.valence:(av?.valence||0);const aro=ps?ps.arousal:(av?.arousal||0);
        // 2026-04-25: lenient demographic binding. `df.age` may be a dict
        // {value, range} (post-process) or a bare int (raw DeepFace).
        // Same for gender/race -- string vs {dominant, scores}. Resolve
        // whichever shape is present, then fall back to `av.*` (set
        // server-side as a safety net), then '?'.
        const _ageVal=df?.age?.range || df?.age?.value || (typeof df?.age==='number'?df.age:null) || av?.age_range || av?.age;
        const _gVal=df?.gender?.dominant || (typeof df?.gender==='string'?df.gender:null) || av?.gender;
        const _rVal=df?.race?.dominant || (typeof df?.race==='string'?df.race:null) || av?.race || av?.ethnicity;
        // v9.7.1 — warmup states. While the value is null, render
        // "warming Xs" with elapsed counter; after _TILE_WARMUP_MS,
        // render "FAILED" so the user knows the tile isn't merely
        // initialising any more. _warmupReset() flips the clock when a
        // real value lands, so a *later* drop-out reuses the warmup
        // grace period instead of immediately crying FAILED.
        const _hasDemog = (_ageVal != null) || _gVal || _rVal;
        if(_hasDemog) _warmupReset(i);
        const ageR = _ageVal != null ? String(_ageVal) : _warmupRender(i, '?');
        const gD   = _gVal || _warmupRender(i, '?');
        const rD   = _rVal || _warmupRender(i, '?');
        const gS=df?.gender?.scores || av?.gender_scores || {};
        const rS=df?.race?.scores   || av?.race_scores   || av?.ethnicity_scores || {};
        // v9.10: blendshape fallback removed from emotion chain
        const faceEmo=f.fused_emotion||df?.emotion?.fused_scores||{};
        let emoBars='';
        if(Object.keys(faceEmo).length){
            emoBars=Object.entries(faceEmo).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([k,v])=>
                '<div class="br"><div class="br-t"><span>'+k+'</span><span>'+Math.round(v*100)+'%</span></div><div class="br-k"><div class="br-f" style="width:'+v*100+'%;background:'+(ec[k]||'#555')+'"></div></div></div>'
            ).join('');
        }
        let gB='';if(Object.keys(gS).length){gB=Object.entries(gS).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<div class="br"><div class="br-t"><span>'+k+'</span><span>'+Math.round(v*100)+'%</span></div><div class="br-k"><div class="br-f" style="width:'+v*100+'%;background:'+(k.toLowerCase().includes('woman')?'#ec4899':'#3b82f6')+'"></div></div></div>').join('');}
        let rB='';const rC={'white':'#94a3b8','black':'#78716c','asian':'#fbbf24','indian':'#f97316','middle eastern':'#a78bfa','latino hispanic':'#34d399'};
        if(Object.keys(rS).length){rB=Object.entries(rS).sort((a,b)=>b[1]-a[1]).map(([k,v])=>'<div class="br"><div class="br-t"><span>'+k+'</span><span>'+Math.round(v*100)+'%</span></div><div class="br-k"><div class="br-f" style="width:'+v*100+'%;background:'+(rC[k.toLowerCase()]||'#71717a')+'"></div></div></div>').join('');}
        const faceLabel=(ps&&ps.identity)?ps.identity:(f.identity||f.df?.identity||f.av?.identity||('FACE '+(i+1)));
        const _cid='fc_'+i;
        cards+='<div class="card"><div class="ct" style="cursor:pointer" onclick="var b=document.getElementById(\''+_cid+'\');b.style.display=b.style.display===\'none\'?\'block\':\'none\'">'+faceLabel.toUpperCase()+' <span style="font-size:7px;color:#71717a">[click]</span></div><div id="'+_cid+'">';
        cards+='<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px"><div class="emo-dot" style="background:'+cl+'"></div><span style="font-size:11px;font-weight:700;color:'+cl+';text-transform:uppercase">'+d_emo+'</span></div>';
        // Per-person vitals (v7). v9.4.1 step 5: apply spoof-mask hatch
        // when PAD verdict is SPOOF with hysteresis. Values stay in the
        // DOM (forensic audit) but are visually overlaid by the mask.
        if(ps){
            // Record the verdict so the hysteresis gate can open/close
            const _padInfo = ps.pad || (d.persons && d.persons[i] && d.persons[i].pad);
            const _padPid = ps.track_id !== undefined ? ps.track_id : i;
            if(_padInfo && _padInfo.verdict){ padRecordVerdict(_padPid, _padInfo.verdict); }
            const _spoofMasked = _padInfo && padShouldMask(_padPid, _padInfo.verdict);
            const _maskCls = _spoofMasked ? ' pad-spoof-mask' : '';
            const _maskAttr = _spoofMasked ? ' data-pad-label="SPOOF"' : '';
            const _hrSrc=ps.hr_source||d.av?.hr_source||'';
            const _srcTag=_hrSrc?' <span style="font-size:7px;color:#71717a">['+_hrSrc+']</span>':'';
            cards+='<div class="g g3'+_maskCls+'"'+_maskAttr+' style="margin-bottom:3px">';
            cards+='<div class="gi"><div style="font-family:monospace;font-size:13px;font-weight:700;color:#ef4444">'+(ps.heart_rate_bpm||'--')+_srcTag+'</div><div class="gl">HR</div></div>';
            cards+='<div class="gi"><div style="font-family:monospace;font-size:13px;font-weight:700;color:#3b82f6">'+(ps.hrv_rmssd?ps.hrv_rmssd.toFixed(0):'--')+'</div><div class="gl">HRV</div></div>';
            cards+='<div class="gi"><div style="font-family:monospace;font-size:13px;font-weight:700;color:#ef4444">'+(ps.stress_index!=null?ps.stress_index.toFixed(2):'--')+'</div><div class="gl">Stress</div></div>';
            cards+='</div>';
            // v8.3: rPPG quality bar (0-1 SNR-based)
            const rq=ps.rppg_quality||0;
            const rqPct=Math.round(rq*100);
            const rqFull=Math.round(rq*5);
            const rqBar=String.fromCharCode(9619).repeat(rqFull)+String.fromCharCode(9617).repeat(5-rqFull);
            const rqCol=rq>0.6?'#22c55e':rq>0.3?'#f59e0b':'#ef4444';
            cards+='<div style="font-family:monospace;font-size:8px;color:'+rqCol+';padding:0 4px">rPPG: '+rqBar+' '+rqPct+'%</div>';
            // Head / Gaze
            if(ps.head_pose){
                cards+='<div class="g g4" style="margin-bottom:3px">';
                cards+='<div class="gi"><div style="font-family:monospace;font-size:11px;color:#f97316">'+ps.head_pose.pitch+'</div><div class="gl">P</div></div>';
                cards+='<div class="gi"><div style="font-family:monospace;font-size:11px;color:#f97316">'+ps.head_pose.yaw+'</div><div class="gl">Y</div></div>';
                cards+='<div class="gi"><div style="font-family:monospace;font-size:11px;color:#ec4899">'+(ps.blink_rate||0)+'</div><div class="gl">Blink</div></div>';
                cards+='<div class="gi"><div style="font-family:monospace;font-size:11px;color:#22d3ee">'+(ps.pupil_dilation||'--')+'</div><div class="gl">Pupil</div></div>';
                cards+='</div>';
            }
            // Deception mini
            if(ps.deception&&ps.deception.baselined){
                const ds=ps.deception.score||0;
                const dc=ds<0.25?'#22c55e':ds<0.5?'#f59e0b':'#ef4444';
                const _dCls = _spoofMasked ? ' pad-spoof-mask' : '';
                const _dAttr = _spoofMasked ? ' data-pad-label="SPOOF"' : '';
                cards+='<div class="'+(_dCls.trim())+'"'+_dAttr+' style="display:flex;align-items:center;gap:4px;padding:2px 4px;background:#18181b;border-radius:3px;margin-bottom:3px;position:relative">';
                cards+='<span style="font-size:8px;color:#71717a" title="Composite of multimodal psychophysiological indicators — research aggregate, not a deception verdict">PSYCHOPHYS</span>';
                cards+='<span style="font-family:monospace;font-size:14px;font-weight:800;color:'+dc+'">'+ds.toFixed(2)+'</span>';
                cards+='<div style="flex:1;height:3px;background:#27272a;border-radius:1px;overflow:hidden"><div style="height:100%;width:'+(ds*100)+'%;background:'+dc+';border-radius:1px"></div></div>';
                cards+='</div>';
            }
        }else{
            cards+='<div class="g g2" style="margin-bottom:3px"><div class="gi"><div style="font-family:monospace;font-size:14px;font-weight:700;color:'+(val>=0?'#22c55e':'#ef4444')+'">'+(val>=0?'+':'')+val.toFixed(2)+'</div><div class="gl">Valence</div></div><div class="gi"><div style="font-family:monospace;font-size:14px;font-weight:700;color:'+(aro>0.5?'#f59e0b':'#3b82f6')+'">'+aro.toFixed(2)+'</div><div class="gl">Arousal</div></div></div>';
        }
        if(emoBars)cards+='<div style="font-size:7px;color:#52525b;margin-bottom:1px;text-transform:uppercase;letter-spacing:1px">EMOTION</div>'+emoBars;
        cards+='<div style="background:#18181b;border-radius:4px;padding:4px;margin-bottom:3px"><div class="g g3"><div style="text-align:center"><div style="font-family:monospace;font-size:11px;font-weight:700;color:#f59e0b">'+ageR+'</div><div class="gl">Age</div></div><div style="text-align:center"><div style="font-size:9px;font-weight:700;color:'+(gD.toLowerCase().includes('woman')?'#ec4899':'#3b82f6')+'">'+gD+'</div><div class="gl">Gender</div></div><div style="text-align:center"><div style="font-size:8px;font-weight:700;color:#a78bfa">'+rD+'</div><div class="gl">Ethnicity</div></div></div>'+gB+rB+'</div>';
        // LLM analysis snippet per person. v9.4.1 step 5: mask when
        // SPOOF verdict latched with hysteresis.
        if(ps&&ps.llm_analysis){
            const _padInfo2 = ps.pad || (d.persons && d.persons[i] && d.persons[i].pad);
            const _padPid2 = ps.track_id !== undefined ? ps.track_id : i;
            const _spoofMasked2 = _padInfo2 && padShouldMask(_padPid2, _padInfo2.verdict);
            const _lCls = _spoofMasked2 ? ' pad-spoof-mask' : '';
            const _lAttr = _spoofMasked2 ? ' data-pad-label="SPOOF"' : '';
            cards+='<div class="'+(_lCls.trim())+'"'+_lAttr+' style="position:relative;font-size:9px;color:#a1a1aa;max-height:50px;overflow:hidden;border-top:1px solid #27272a;padding-top:3px;margin-top:2px">'+ps.llm_analysis.substring(0,200)+'</div>';
        }
        cards+='</div></div>';  // close collapsible div + card
    });
    if(_cache['fc']!==cards){_cache['fc']=cards;fc.innerHTML=cards;}
}
function uAI(m){
// GUARD: never overwrite truth test UI with MON messages
if(truthState!=='idle')return;
const el=document.getElementById('aiBox');
// Cache on TEXT content only (not timestamps) to prevent twitching
const k=JSON.stringify(m?.map(x=>x.text)||'');
if(_cache['ai']===k)return;_cache['ai']=k;
if(!m||!m.length){el.innerHTML='<div class="ai-e">No analysis</div>';return;}
document.getElementById('aiTime').textContent='Last: '+m[m.length-1].t;
el.innerHTML=m.map(x=>'<div class="ai-msg"><div class="ai-t">'+x.t+'</div><div class="ai-x">'+x.text+'</div></div>').join('');}

// Step 8: Deception Dashboard
const DEC_INDICATORS=['pupil_dilation','blink_suppression','duchenne_smile','facial_asymmetry','fear_leakage','heart_rate_change','vocal_f0_shift','response_latency','micro_expression','gsr_change','hrv_drop'];
function uDeception(p){
  if(!p||!p.deception)return;
  const d=p.deception,score=d.score||0;
  const el=document.getElementById('deceptionScore');
  el.textContent=score.toFixed(3);
  el.style.color=score<0.25?'#22c55e':score<0.5?'#f59e0b':'#ef4444';
  let html='';
  if(d.indicators){
    for(const k of DEC_INDICATORS){
      const v=d.indicators[k]||0;
      const c=v<0.25?'#22c55e':v<0.5?'#f59e0b':'#ef4444';
      const label=k.replace(/_/g,' ');
      html+='<div style="display:flex;align-items:center;gap:3px;margin:1px 0"><span style="width:85px;text-align:right;color:#52525b">'+label+'</span>';
      html+='<div style="flex:1;height:3px;background:#18181b;border-radius:1px;overflow:hidden"><div style="height:100%;width:'+(v*100)+'%;background:'+c+';border-radius:1px;transition:width 0.4s"></div></div>';
      html+='<span style="width:28px;text-align:right;color:#71717a">'+(v*100).toFixed(0)+'%</span></div>';
    }
  }
  document.getElementById('deceptionBars').innerHTML=html;
}
// Step 9: Frame Quality
function uQuality(q){
  const el=document.getElementById('qualityWarn');
  if(!q||q.overall_ok){el.style.display='none';return;}
  el.style.display='block';el.textContent='QUALITY: '+(q.warnings||[]).join(' | ');
}
// v8.2: Governor token budget gauge + diarizer speaker count
function uGovBudget(s){
  const gov=s?.llm_governor;
  if(!gov)return;
  const used=gov.hourly_tokens_used||0;
  const budget=gov.hourly_tokens_budget||10000;
  const pct=Math.min(100,Math.round(100*used/budget));
  const bc=pct>80?'#ef4444':pct>50?'#f59e0b':'#22c55e';
  sEl('sGovBudget','GOV: '+used+'/'+budget,bc);
  const bar=document.getElementById('govBar');
  if(bar){bar.style.width=pct+'%';bar.style.background=bc;}
}
function uSpeakers(d){
  const n=d?.av?.diarize_speakers;
  if(n!=null){sEl('sSpeakers','SPK: '+n,n>0?'#22c55e':'#52525b');}
}

// v9.5.x — `skipLandmarks` = true means the GPU tier (WebGL/WebGPU) already
// drew the landmark mesh; skip the 478×N-per-face 2D arc loop and only do
// overlay annotations (PAD ring, iris, gaze, pose). Saves 5-15 ms / frame.
function draw(d, skipLandmarks){
    c.clearRect(0,0,W,H);
    // v9.7.4: tick the mesh-render counter exactly once per draw call,
    // before the per-face loop reads it. Counter wraps at a large
    // boundary to avoid drift over a long session.
    _meshRenderTick = (_meshRenderTick + 1) & 0x7fffffff;
    // v9.4 step 9 — INSTANT-tier PAD ring. Read per-person PAD verdict
    // from d.persons[] → pad.verdict; paint a coloured bbox ring around
    // that face. Green=LIVE, Amber=UNCERTAIN, Red=SPOOF. The ring is
    // drawn BEFORE the landmarks so the landmarks sit on top.
    const _padByPid = (()=>{ const m = new Map();
        for(const p of (d.persons||[])){
            if(p && p.pad && p.track_id !== undefined) m.set(p.track_id, p.pad);
        } return m; })();
    for(const fIdx in (d.faces||[])){
        const f=d.faces[fIdx];
        const df=f.df||dfc[f.i];
        // v9.7.2: per-face dominant emotion. Priority:
        //   1. f.fused_emotion (per-face HSEmotion+blendshape, multi-face fixed in main.py)
        //   2. per-face DeepFace cache (df.emotion.dominant)
        //   3. matching person's tracked dominant_emotion (persons[fIdx])
        //   4. global d.av.dominant_emotion (single-person summary)
        //   5. per-face raw blendshape argmax
        // Previously this used d.av first, painting every face's ring with
        // the primary face's emotion color. Multi-face scenes now show
        // correct per-face ring colors.
        const _ff=f.fused_emotion;
        const _ffDom=_ff?Object.keys(_ff).reduce((a,b)=>_ff[a]>_ff[b]?a:b,'neutral'):null;
        const _psMatch=(d.persons&&d.persons[fIdx])?d.persons[fIdx]:null;
        // v9.10: blendshape argmax removed from this chain too
        const dm=_ffDom||df?.emotion?.dominant||_psMatch?.dominant_emotion||d.av?.dominant_emotion||'neutral';
        const cl=ec[dm]||'#22c55e';
        // PAD ring: derive a face bbox from its landmarks.
        try {
            if(FF && FF.PAD_UI && f.lm && f.lm.length){
                const padInfo = _padByPid.get(f.i) || _padByPid.get(f.track_id);
                if(padInfo){
                    let minX=1,minY=1,maxX=0,maxY=0;
                    for(const l of f.lm){
                        const lx=Array.isArray(l)?l[0]:l.x, ly=Array.isArray(l)?l[1]:l.y;
                        if(lx<minX)minX=lx; if(lx>maxX)maxX=lx;
                        if(ly<minY)minY=ly; if(ly>maxY)maxY=ly;
                    }
                    const px=minX*W-6, py=minY*H-6, pw=(maxX-minX)*W+12, ph=(maxY-minY)*H+12;
                    const ringCol = padInfo.verdict === 'SPOOF' ? '#ef4444'
                                 : padInfo.verdict === 'UNCERTAIN' ? '#f59e0b'
                                 : '#22c55e';
                    c.save();
                    c.globalAlpha = 0.85;
                    c.strokeStyle = ringCol;
                    c.lineWidth = 3;
                    c.strokeRect(px, py, pw, ph);
                    if(padInfo.verdict === 'SPOOF'){
                        c.globalAlpha = 0.75;
                        c.fillStyle = '#ef4444';
                        c.font = '700 11px Segoe UI';
                        c.fillText('⚠ PAD_FAIL · ' + (padInfo.attack_class||'UNKNOWN'), px+4, py-4);
                    }
                    c.restore();
                }
            }
        } catch(e){}
        // v9.7.4: skip mesh render based on `?meshskip=N` URL param.
        //   meshskip=1 (default) → render every frame
        //   meshskip=2          → every 2nd frame
        //   meshskip=0          → never (PAD ring + iris still draw)
        // Drawing 478 dots × N faces every frame is the dominant cost
        // in the per-frame canvas paint; halving it nearly doubles
        // visible fps on slower machines.
        const _renderMeshThisFrame = (_meshRenderEvery > 0)
            && (_meshRenderTick % _meshRenderEvery === 0);
        if(!skipLandmarks && _renderMeshThisFrame){
            for(const l of f.lm){const lx=Array.isArray(l)?l[0]:l.x,ly=Array.isArray(l)?l[1]:l.y;c.beginPath();c.arc(lx*W,ly*H,1.3,0,6.28);c.fillStyle=cl;c.globalAlpha=0.85;c.fill();}
        }
        c.globalAlpha=1.0;
        const ir=f.iris;
        if(ir){
            if(ir.l_eye&&ir.l_eye.length>2){c.beginPath();c.strokeStyle='#06b6d4';c.lineWidth=1;c.globalAlpha=0.7;ir.l_eye.forEach((p,i)=>{i?c.lineTo(p.x*W,p.y*H):c.moveTo(p.x*W,p.y*H)});c.closePath();c.stroke();}
            if(ir.r_eye&&ir.r_eye.length>2){c.beginPath();c.strokeStyle='#06b6d4';c.lineWidth=1;ir.r_eye.forEach((p,i)=>{i?c.lineTo(p.x*W,p.y*H):c.moveTo(p.x*W,p.y*H)});c.closePath();c.stroke();}
            c.globalAlpha=1.0;
            if(ir.l_ring&&ir.l_ring.length>2){c.beginPath();c.strokeStyle='#22d3ee';c.lineWidth=1.5;ir.l_ring.forEach((p,i)=>{i?c.lineTo(p.x*W,p.y*H):c.moveTo(p.x*W,p.y*H)});c.closePath();c.stroke();}
            if(ir.r_ring&&ir.r_ring.length>2){c.beginPath();c.strokeStyle='#22d3ee';c.lineWidth=1.5;ir.r_ring.forEach((p,i)=>{i?c.lineTo(p.x*W,p.y*H):c.moveTo(p.x*W,p.y*H)});c.closePath();c.stroke();}
            if(ir.l_iris){c.beginPath();c.arc(ir.l_iris.x*W,ir.l_iris.y*H,2.5,0,6.28);c.fillStyle='#22d3ee';c.fill();}
            if(ir.r_iris){c.beginPath();c.arc(ir.r_iris.x*W,ir.r_iris.y*H,2.5,0,6.28);c.fillStyle='#22d3ee';c.fill();}
            if(f.gaze&&ir.l_iris&&ir.r_iris){
                const gx=f.gaze.horizontal*20;const gy=f.gaze.vertical*15;
                c.beginPath();c.strokeStyle='#f97316';c.lineWidth=1;
                c.moveTo(ir.l_iris.x*W,ir.l_iris.y*H);c.lineTo(ir.l_iris.x*W+gx,ir.l_iris.y*H+gy);c.stroke();
                c.beginPath();c.moveTo(ir.r_iris.x*W,ir.r_iris.y*H);c.lineTo(ir.r_iris.x*W+gx,ir.r_iris.y*H+gy);c.stroke();
            }
            if(ir.avg_dilation){
                const dilSize=ir.avg_dilation*30;
                c.beginPath();c.arc(ir.l_iris.x*W,ir.l_iris.y*H,dilSize,0,6.28);c.strokeStyle='rgba(34,211,238,0.3)';c.lineWidth=1;c.stroke();
                c.beginPath();c.arc(ir.r_iris.x*W,ir.r_iris.y*H,dilSize,0,6.28);c.stroke();
            }
        }
        if(f.pose&&f.lm.length>2){
            const _n=f.lm[1]||f.lm[0];const nx=Array.isArray(_n)?_n[0]:_n.x,ny=Array.isArray(_n)?_n[1]:_n.y;
            const ax=Math.sin(f.pose.yaw*Math.PI/180)*25;const ay=-Math.sin(f.pose.pitch*Math.PI/180)*25;
            c.beginPath();c.strokeStyle='#f97316';c.lineWidth=2;
            c.moveTo(nx*W,ny*H);c.lineTo(nx*W+ax,ny*H+ay);c.stroke();
            c.beginPath();c.fillStyle='#f97316';c.arc(nx*W+ax,ny*H+ay,3,0,6.28);c.fill();
        }
        if(f.lm.length>0){const _p=f.lm[0];const px=Array.isArray(_p)?_p[0]:_p.x,py=Array.isArray(_p)?_p[1]:_p.y;c.textAlign='center';
            const id=f.identity||df?.identity;
            if(id){c.font='700 12px Segoe UI';c.fillStyle='#06b6d4';c.fillText(id.toUpperCase(),px*W,py*H-62);}
            c.font='700 10px Segoe UI';c.fillStyle=cl;c.fillText('['+(f.i+1)+'] '+dm.toUpperCase(),px*W,py*H-52);
            if(df){c.font='500 8px Segoe UI';c.fillStyle='#a1a1aa';c.fillText((df.age?.range||'?')+' / '+(df.gender?.dominant||'?'),px*W,py*H-41);
            c.font='500 7px Segoe UI';c.fillStyle='#71717a';c.fillText(df.race?.dominant||'?',px*W,py*H-31);}
            if(ir?.avg_dilation){c.font='500 7px Segoe UI';c.fillStyle='#22d3ee';c.fillText('pupil: '+ir.avg_dilation.toFixed(2),px*W,py*H-21);}
            c.textAlign='left';
        }
    }
}

async function askAI(){document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#a855f7">Analyzing all persons...</div>';
try{
    // Try per-person analysis first (v7)
    const r2=await fetch('/analyze/persons',{method:'POST'});const d2=await r2.json();
    if(!d2.error&&(d2.per_person||d2.room)){
        let txt='';
        if(d2.per_person){for(const[tid,text]of Object.entries(d2.per_person)){txt+='[Person #'+tid+'] '+text+String.fromCharCode(10,10);}}
        if(d2.room){txt+='[ROOM DYNAMICS] '+d2.room;}
        _cache['ai']='';uAI([{t:new Date().toLocaleTimeString(),text:txt}]);
    }else{
        // Fallback to single-frame analysis
        const tmp=document.createElement('canvas');tmp.width=W;tmp.height=H;tmp.getContext('2d').drawImage(v,0,0,W,H);
        const blob=await new Promise(r=>tmp.toBlob(r,'image/jpeg',0.6));const fd=new FormData();fd.append('file',blob,'f.jpg');
        const r=await fetch('/analyze',{method:'POST',body:fd});const d=await r.json();
        if(d.error){document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">'+d.error+'</div>';}
        else{_cache['ai']='';uAI([{t:new Date().toLocaleTimeString(),text:d.analysis}]);}
    }
}catch(e){document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">'+e.message+'</div>';}}

async function toggleMon(){monOn=!monOn;const b=document.getElementById('monBtn');const sp=document.getElementById('aiSpinner');
if(monOn){await fetch('/monitor/start',{method:'POST'});b.textContent='MON: ON';b.style.background='#22c55e';b.style.borderColor='#16a34a';sp.style.display='inline-block';}
else{await fetch('/monitor/stop',{method:'POST'});b.textContent='MON: OFF';b.style.background='#f59e0b';b.style.borderColor='#d97706';sp.style.display='none';}}

// v8.7 FINAL BUGFIX: completely rewritten camera startup. The root cause of
// the "START button does nothing" symptom was a three-way race between:
//   (a) a 300ms auto-start firing before any user gesture (Chrome may defer
//       the permission prompt for non-gesture calls, leaving state
//       half-initialized);
//   (b) a spacebar keydown handler that ALSO called go();
//   (c) the button's own onclick="go()".
// Any two of those overlapping could deadlock getUserMedia into
// NotReadableError on the second call, and because `on` only flipped true
// at the very end of go(), the re-entrancy guard alone couldn't save us.
//
// New design:
//   * A single re-entrancy guard (`_starting`) flipped SYNCHRONOUSLY.
//   * ONE entry point: the START button. No auto-start, no keydown, no
//     window-level listeners.
//   * Every step prints a console log AND updates the visible status line
//     so diagnosis is instant when things go wrong.
//   * Structured error → hint mapping surfaced directly to the user.
let _starting=false;
let _clickCount=0;
async function go(){
    _clickCount++;
    try{window.__diag('START clicked ('+_clickCount+'x) on='+on+' _starting='+_starting,'#ff0');}catch(_){}
    const dbg=document.getElementById('stats');
    const setStatus=(txt,col)=>{
        try{dbg.textContent=txt;if(col)dbg.style.color=col;}catch(_){}
        try{document.getElementById('sWcam').textContent='CAM: '+txt;
            if(col)document.getElementById('sWcam').style.color=col;}catch(_){}
        try{window.__diag('stage: '+txt,col||'#0f0');}catch(_){}
        console.log('[CAM]',txt);
    };
    if(on){setStatus('Already running','#22c55e');return;}
    if(_starting){try{window.__diag('go() already in flight, skipping','#ff0');}catch(_){}return;}
    _starting=true;
    try{
        if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
            setStatus('CAMERA UNAVAILABLE — use http://127.0.0.1:8000/webcam (localhost or HTTPS required by the browser)','#ef4444');
            return;
        }
        if(!window.isSecureContext){
            setStatus('CAMERA BLOCKED — non-secure context ('+location.origin+'). Open http://127.0.0.1:8000/webcam','#ef4444');
            return;
        }
        setStatus('Requesting camera permission...','#f59e0b');
        // Probe device list (does not require permission in most browsers)
        try{
            const _devs=await navigator.mediaDevices.enumerateDevices();
            const _cams=_devs.filter(d=>d.kind==='videoinput');
            window.__diag('devices: '+_cams.length+' camera(s) detected','#0ff');
            _cams.forEach((d,i)=>window.__diag('  cam['+i+']: '+(d.label||'(no label — permission not granted yet)'),'#0ff'));
            if(_cams.length===0){
                setStatus('NO CAMERA DEVICES FOUND — check Windows Settings > Privacy > Camera and unplug/replug','#ef4444');
                return;
            }
        }catch(_de){window.__diag('enumerateDevices failed: '+_de.message,'#fa0');}
        // v9.7.3: avoid colliding with the dual_camera high-fps path.
        // The server holds whichever camera matches `_HIFPS_PREFERRED_NAME_PATTERNS`
        // in `dual_camera.py` (currently "HD USB Camera"). If the
        // browser's default `getUserMedia({video:true})` lands on the
        // same camera, we get `NotReadableError: Device in use`. Strategy:
        //   1. Enumerate devices first (labels populate after any prior
        //      permission grant or after a failed-but-prompted call).
        //   2. Identify candidates whose label DOESN'T match a server-
        //      reserved pattern.
        //   3. Try them in order with explicit deviceId.
        //   4. Fall back to default if no labels are known yet (first
        //      run, no permission), then catch NotReadableError and
        //      retry against the OTHER deviceId.
        // Patterns kept in sync with dual_camera._HIFPS_PREFERRED_NAME_PATTERNS.
        const SERVER_CAM_PATTERNS = ['hd usb camera','elp','svpro','120fps','120 fps','high speed'];
        // v9.7.4: virtual / software-only camera apps that should NEVER
        // be picked by `getUserMedia()`. Mirrors `dual_camera._VIRTUAL_CAM_PATTERNS`
        // on the server so both sides agree about what counts as a
        // virtual cam. The earlier UI bug: when the server held the
        // physical cam, the retry loop fell through to "OBS Virtual
        // Camera" because it wasn't in SERVER_CAM_PATTERNS, but it's
        // also not a real camera if OBS isn't running — so getUserMedia
        // failed with NotReadableError and the user saw the bogus
        // `(Zoom/Teams/Meet/OBS?)` hint even though no app was holding
        // the cam, the retry just chose a non-existent virtual one.
        const VIRTUAL_CAM_PATTERNS = [
            'obs ','obs virtual','obs-virtual','virtual camera','virtualcam','vcam',
            'snap camera','snapcam','streamlabs','xsplit','scarlet',
            'ndi ','ndi-','manycam','ecamm','epoccam','droidcam'
        ];
        const isServerCam = (label) => {
            const l = (label || '').toLowerCase();
            if (!l) return false;
            return SERVER_CAM_PATTERNS.some(p => l.includes(p));
        };
        const isVirtualCam = (label) => {
            const l = (label || '').toLowerCase();
            if (!l) return false;
            return VIRTUAL_CAM_PATTERNS.some(p => l.includes(p));
        };
        // A "browser cam" = physical, NOT server-reserved, NOT virtual.
        const isBrowserCam = (label) => {
            return !isServerCam(label) && !isVirtualCam(label);
        };
        const PREFERRED_BROWSER_CAM_PATTERNS = ['eos webcam utility','canon webcam utility','canon eos','elgato cam link','magewell','blackmagic'];
        const _DODO_LAUNCHER_MODE = ((window.DODO_CAMERA_MODE || 'AUTO') + '').toUpperCase();
        let _DODO_EFFECTIVE_MODE = _DODO_LAUNCHER_MODE;
        const buildCfg = (deviceId, hd) => {
            const isTri = (_DODO_EFFECTIVE_MODE === 'TRI');
            const v = hd
                ? (isTri
                    ? {width:{ideal:1920,max:1920},height:{ideal:1080,max:1080},frameRate:{ideal:60,max:60}}
                    : {width:{ideal:1280,max:1920},height:{ideal:720,max:1080},frameRate:{ideal:30,max:60}})
                : {width:{ideal:640},height:{ideal:480},frameRate:{ideal:30,max:60}};
            if (deviceId) v.deviceId = {exact: deviceId};
            else          v.facingMode = 'user';
            return {video:v};
        };

        // Enumerate; pick non-server, non-virtual cam if labels are populated.
        // v9.8: permission warmup. Browsers (Chrome/Edge) return
        // `label=""` from enumerateDevices() until at least one
        // getUserMedia() call has succeeded -- so the smart picker below
        // sees no labels on FIRST run and falls through to default,
        // which often grabs the high-fps cam, locking the server out
        // of its rPPG path. Fix: do a tiny throwaway getUserMedia() to
        // grant permission, then enumerate again (labels now populated),
        // then pick the right cam, and only NOW start the real capture.
        let preferredId = null;
        try {
            // v9.8 step 1: enumerate first (cheap) -- on subsequent visits
            // the labels are already cached, no warmup needed.
            let _devs = await navigator.mediaDevices.enumerateDevices();
            let _cams = _devs.filter(d => d.kind === 'videoinput');
            const _haveLabels = _cams.some(c => c.label);
            if (!_haveLabels && _cams.length > 0) {
                // v9.8 step 2: trigger permission so labels populate. We
                // intentionally request the LOWEST possible config to make
                // the prompt fast; the stream is closed immediately after.
                window.__diag('label warmup: requesting any-cam permission to populate labels...','#0ff');
                try {
                    const _warm = await navigator.mediaDevices.getUserMedia({video:{width:160,height:120}});
                    _warm.getTracks().forEach(t => t.stop());
                } catch(_we) {
                    window.__diag('warmup getUserMedia failed: '+_we.name+' -- continuing','#fa0');
                }
                _devs = await navigator.mediaDevices.enumerateDevices();
                _cams = _devs.filter(d => d.kind === 'videoinput');
            }
            window.__diag('post-warmup devices: '+_cams.length+' camera(s)','#0ff');
            _cams.forEach((d,i)=>window.__diag('  cam['+i+']: '+(d.label||'(still no label)'),'#0ff'));

            // v9.8 step 3: now pick smart. Server-reserved cams
            // (HD USB Camera / ELP / etc.) are excluded so the server
            // gets exclusive 120 fps access. Within browser-eligible
            // cams, prefer EOS (when allowed) > UGREEN > non-integrated > integrated.
            const browserCams = _cams.filter(c => isBrowserCam(c.label));
            const isPreferred = (lbl) => {
                const l = (lbl || '').toLowerCase();
                return PREFERRED_BROWSER_CAM_PATTERNS.some(p => l.indexOf(p) >= 0);
            };
            const _eosPriority = (lbl) => {
                const l = (lbl || '').toLowerCase();
                if (l.indexOf('eos webcam utility pro') >= 0) return 0;
                if (l.indexOf('canon eos r10') >= 0)          return 1;
                if (l.indexOf('elgato cam link') >= 0)        return 1;
                if (l.indexOf('magewell') >= 0)               return 1;
                if (l.indexOf('blackmagic') >= 0)             return 1;
                if (l.indexOf('eos webcam utility') >= 0)     return 2;
                if (l.indexOf('canon webcam utility') >= 0)   return 2;
                return 3;
            };
            let preferredSorted;
            if (_DODO_LAUNCHER_MODE === 'DUAL') {
                preferredSorted = [];
            } else {
                preferredSorted = browserCams
                    .filter(c => isPreferred(c.label))
                    .sort((a, b) => _eosPriority(a.label) - _eosPriority(b.label));
            }
            const r10pro = preferredSorted[0];
            const ugreen = browserCams.find(c => /ugreen/i.test(c.label));
            const integrated = browserCams.find(c => /integrated|laptop|built|hp\s|lenovo|dell\s/i.test(c.label));
            const otherNonIntegrated = browserCams.find(c =>
                !isPreferred(c.label) && !/ugreen/i.test(c.label)
                && !/integrated|laptop|built|hp\s|lenovo|dell\s/i.test(c.label));
            if (_DODO_LAUNCHER_MODE === 'AUTO') {
                const _reservedServer = _cams.some(c => isServerCam(c.label));
                if (r10pro && (ugreen || _reservedServer)) _DODO_EFFECTIVE_MODE = 'TRI';
                else if (ugreen || otherNonIntegrated) _DODO_EFFECTIVE_MODE = 'DUAL';
                else _DODO_EFFECTIVE_MODE = 'SINGLE';
            } else if (_DODO_LAUNCHER_MODE === 'TRI' || _DODO_LAUNCHER_MODE === 'DUAL') {
                _DODO_EFFECTIVE_MODE = _DODO_LAUNCHER_MODE;
            }
            const _pick = r10pro || ugreen || otherNonIntegrated || integrated || browserCams[0];
            if (_pick && _pick.deviceId) {
                preferredId = _pick.deviceId;
                window.__diag('PICKED for browser HQ face stream: '+_pick.label,'#0f0');
                const reserved = _cams.filter(c => isServerCam(c.label));
                if (reserved.length) {
                    window.__diag('  reserved for server 120fps: '+reserved.map(c=>c.label).join(', '),'#0ff');
                }
            } else if (_cams.length === 1 && _cams[0].deviceId && _cams[0].label
                       && !isVirtualCam(_cams[0].label)) {
                preferredId = _cams[0].deviceId;
                window.__diag('only one cam available: '+_cams[0].label,'#fa0');
            }
        } catch(_de) {
            window.__diag('enumerate/pick failed: '+_de.message+' -- letting browser pick default','#fa0');
        }

        try{
            window.__diag('calling getUserMedia(HD'+(preferredId?', deviceId':'')
                +', mode='+_DODO_EFFECTIVE_MODE+'; launcher='+_DODO_LAUNCHER_MODE+')...','#0ff');
            st = await navigator.mediaDevices.getUserMedia(buildCfg(preferredId, true));
            window.__diag('getUserMedia(HD) OK tracks='+st.getVideoTracks().length,'#0f0');
        }
        catch(_e1){
            window.__diag('HD failed: '+_e1.name+' — '+_e1.message,'#fa0');
            // If NotReadableError on default, the server probably has it.
            // Re-enumerate (permission may now be granted from the failed
            // attempt) and try the OTHER camera explicitly.
            if (_e1.name === 'NotReadableError' || _e1.name === 'OverconstrainedError') {
                try {
                    const _devs2 = await navigator.mediaDevices.enumerateDevices();
                    const _cams2 = _devs2.filter(d => d.kind === 'videoinput');
                    // v9.7.4: retry must also exclude virtual cams. The
                    // earlier bug was the retry loop falling through to
                    // OBS Virtual Camera (because it wasn't in
                    // SERVER_CAM_PATTERNS) and getting NotReadableError
                    // again because OBS wasn't running.
                    const browserCams2 = _cams2.filter(c => isBrowserCam(c.label));
                    let target = browserCams2[0];
                    if (!target) {
                        // Fallback: any cam that isn't the preferred one
                        // AND isn't virtual. Better than blindly trying
                        // OBS again.
                        target = _cams2.find(c =>
                            c.deviceId !== preferredId && !isVirtualCam(c.label));
                    }
                    if (target && target.deviceId) {
                        window.__diag('retrying with non-default cam: '+(target.label||target.deviceId),'#fa0');
                        st = await navigator.mediaDevices.getUserMedia(buildCfg(target.deviceId, true));
                        window.__diag('retry OK','#0f0');
                    } else {
                        window.__diag('no non-virtual cam available for retry','#fa0');
                    }
                } catch(_eR) {
                    window.__diag('retry failed: '+_eR.name+' — '+_eR.message,'#fa0');
                }
            }
            if (!st) {
                try{
                    window.__diag('calling getUserMedia(SD)...','#0ff');
                    st = await navigator.mediaDevices.getUserMedia(buildCfg(preferredId, false));
                    window.__diag('getUserMedia(SD) OK','#0f0');
                }
                catch(_e2){
                    window.__diag('SD failed: '+_e2.name+' — '+_e2.message,'#fa0');
                    window.__diag('calling getUserMedia({video:true})...','#0ff');
                    st = await navigator.mediaDevices.getUserMedia({video:true});
                    window.__diag('getUserMedia(bare) OK','#0f0');
                }
            }
        }
        setStatus('Camera acquired, starting video element...','#f59e0b');
        try{
            const _track=st.getVideoTracks()[0];
            const _set=_track?_track.getSettings():{};
            window.__diag('track: '+(_track&&_track.label||'?')+' '+(_set.width||'?')+'x'+(_set.height||'?')+'@'+(_set.frameRate||'?')+'fps','#0f0');
        }catch(_te){}
        v.srcObject=st;
        v.muted=true;v.playsInline=true;v.autoplay=true;
        // v9.7.3 BUGFIX: sync W/H + overlay + capture canvas to actual
        // camera resolution. Was: W=640, H=480 hardcoded at boot. With
        // an HD camera producing 1280×720, the source got non-uniformly
        // squished into a 4:3 box (16:9 → 4:3 means 0.5× X, 0.667× Y),
        // and any subsequent layer that touched intrinsic dimensions
        // (CSS box vs canvas drawing surface, client-tracker referencing
        // videoEl native coords vs server-normalised on the squished
        // capture canvas) introduced a vertical drift that pushed the
        // mesh dots off the actual face. Now: internal coords match
        // native video resolution; CSS scales the display down to a
        // fixed visual width while preserving aspect ratio; overlay
        // canvas always overlays the video pixel-for-pixel.
        const _DISPLAY_W = 640;
        const _syncVideoSize = () => {
            if(!v.videoWidth || !v.videoHeight) return;
            // v9.7.4 PERF: cap the **internal canvas drawing surface**
            // at the display width regardless of native video resolution.
            // Previous version set canvas.width = videoWidth (1280) which
            // quadrupled the per-frame work — every clearRect cleared
            // 921k pixels instead of 307k, every PAD ring/landmark draw
            // operation worked at 2× linear scale → canvas paint dropped
            // from ~30 fps to ~16 fps.
            //
            // Landmark normalised coords (0-1) map correctly to ANY
            // canvas size, so reducing the drawing surface to the
            // display size is lossless for our overlays. The capture
            // canvas (`_capCanvas`) DOES stay at native res because the
            // server-side MediaPipe + DeepFace benefit from full
            // resolution; only the display overlay is capped.
            const _native_w = v.videoWidth;
            const _native_h = v.videoHeight;
            const _dispH = Math.round(_DISPLAY_W * _native_h / _native_w);
            // W/H globals = display surface size, used by every draw call
            W = _DISPLAY_W;
            H = _dispH;
            if(_overlayEl){
                _overlayEl.width  = W;
                _overlayEl.height = H;
                _overlayEl.style.width  = _DISPLAY_W + 'px';
                _overlayEl.style.height = _dispH + 'px';
            }
            // Capture canvas keeps native resolution — the JPEG that
            // goes server-side is full quality. Server-side coords are
            // normalised 0-1 so they re-map correctly to the smaller
            // overlay W,H.
            if(_capCanvas){
                _capCanvas.width  = _native_w;
                _capCanvas.height = _native_h;
            }
            v.style.width  = _DISPLAY_W + 'px';
            v.style.height = _dispH + 'px';
            v.style.objectFit = 'fill';
            window.__diag('video native '+_native_w+'×'+_native_h+
                          ', overlay/display '+W+'×'+H,'#0f0');
        };
        v.addEventListener('loadedmetadata', _syncVideoSize);
        if(v.videoWidth) _syncVideoSize();  // already loaded
        try{ await v.play(); window.__diag('v.play() resolved','#0f0'); }
        catch(_pe){ window.__diag('v.play() rejected: '+_pe.name+' — '+_pe.message,'#fa0'); }
        // After play() the metadata is guaranteed loaded — sync once more
        // in case the listener missed the (already-fired) event.
        _syncVideoSize();
        // Wait until at least one frame is decoded, with a hard cap
        await new Promise(res=>{
            if(v.readyState>=2){window.__diag('video.readyState='+v.readyState+' (already decoded)','#0f0');res();return;}
            window.__diag('waiting for loadeddata (readyState='+v.readyState+')...','#0ff');
            const _t=setTimeout(()=>{window.__diag('loadeddata TIMED OUT at 2s (readyState='+v.readyState+')','#fa0');res();},2000);
            v.addEventListener('loadeddata',()=>{clearTimeout(_t);window.__diag('loadeddata fired','#0f0');res();},{once:true});
        });
        setStatus('Video playing — opening WebSocket...','#22c55e');
        on=true;n=0;dfc={};lastAV={};fpsArr=[];
        try{const _b=document.getElementById('startBtn');if(_b)_b.classList.remove('start-attention');}catch(_){}
        fetch('/vitals/reset',{method:'POST'}).catch(()=>{});
        fetch('/status').then(r=>r.ok?r.json():null).then(s=>{
            if(!s)return;
            try{
                document.getElementById('sLlm').textContent='LLM: '+(s.llm?.backend||'?').toUpperCase();
                document.getElementById('sLlm').style.color=s.llm?.connected?'#22c55e':'#f59e0b';
                document.getElementById('sFaces').textContent='FACES: '+s.faces;
                document.getElementById('sFaces').style.color=s.faces>0?'#22c55e':'#52525b';
                uGovBudget(s);
            }catch(_e){}
        }).catch(()=>{});
        startWS();
    }catch(e){
        on=false;
        let hint='';
        if(e.name==='NotAllowedError'||e.name==='SecurityError') hint=' — click the camera icon in the address bar and ALLOW, then click START again';
        else if(e.name==='NotFoundError'||e.name==='OverconstrainedError') hint=' — no compatible webcam detected on this machine';
        else if(e.name==='NotReadableError'||e.name==='AbortError') hint=' — camera in use by another app (Zoom/Teams/Meet/OBS?). Close it and click START again';
        else if(e.name==='TypeError') hint=' — browser security: open via http://127.0.0.1:8000/webcam (not file:// or remote http)';
        setStatus('CAMERA ERROR ('+(e.name||'?')+'): '+e.message+hint,'#ef4444');
        console.error('[CAM] getUserMedia failed:',e);
    }finally{
        _starting=false;
    }
}
function stop(){
    on=false;
    if(ws){try{ws.close();}catch(_){}ws=null;wsReady=false;}
    if(st){try{st.getTracks().forEach(t=>t.stop());}catch(_){}st=null;}
    try{v.srcObject=null;}catch(_){}
    try{c.clearRect(0,0,W,H);}catch(_){}
    document.getElementById('stats').textContent='Stopped — click START to restart';
    document.getElementById('stats').style.color='#71717a';
    const sw=document.getElementById('sWcam');if(sw){sw.textContent='CAM: off';sw.style.color='#52525b';}
    // Invite another START click with the pulse again
    try{const _b=document.getElementById('startBtn');if(_b)_b.classList.add('start-attention');}catch(_){}
}

// v9.10 fix (2026-05-09): release the camera handle when the user
// closes / navigates away from the tab. Without this, the browser
// keeps the MediaStreamTrack alive past tab close (especially on
// Chromium where tabs frequently background-suspend before fully
// dying), and the next dodo session sees `NotReadableError -- Device
// in use` until the user manually opens Windows Camera Settings (the
// observed v9.9 bug). pagehide fires more reliably than beforeunload
// on mobile + bfcache, so we hook both.
function _releaseCameraOnClose(){
    try{
        if(typeof st !== 'undefined' && st){
            st.getTracks().forEach(t => { try{ t.stop(); }catch(_){} });
        }
        if(typeof v !== 'undefined' && v){ try{ v.srcObject=null; }catch(_){} }
    }catch(_){}
}
window.addEventListener('beforeunload', _releaseCameraOnClose);
window.addEventListener('pagehide', _releaseCameraOnClose);
function rs(){fetch('/vitals/reset',{method:'POST'});document.getElementById('hrL').textContent='--';document.getElementById('hrC').textContent='--';}

// Face registration functions
async function regFace(){
    const name=document.getElementById('faceName').value.trim();
    if(!name){document.getElementById('faceStatus').textContent='Enter a name first';return;}
    document.getElementById('faceStatus').textContent='Capturing...';
    const tmp=document.createElement('canvas');tmp.width=W;tmp.height=H;tmp.getContext('2d').drawImage(v,0,0,W,H);
    const blob=await new Promise(r=>tmp.toBlob(r,'image/jpeg',0.9));
    const fd=new FormData();fd.append('file',blob,'face.jpg');fd.append('name',name);
    try{const r=await fetch('/faces/register',{method:'POST',body:fd});const d=await r.json();
    document.getElementById('faceStatus').textContent=d.ok?'Learned: '+name:'Error: '+(d.error||'failed');
    document.getElementById('faceName').value='';}catch(e){document.getElementById('faceStatus').textContent='Error: '+e.message;}
}
function regPhoto(){document.getElementById('faceFile').click();}
async function uploadFaceFile(){
    const name=document.getElementById('faceName').value.trim();
    if(!name){document.getElementById('faceStatus').textContent='Enter a name first';return;}
    const file=document.getElementById('faceFile').files[0];if(!file)return;
    document.getElementById('faceStatus').textContent='Uploading...';
    const fd=new FormData();fd.append('file',file);fd.append('name',name);
    try{const r=await fetch('/faces/register',{method:'POST',body:fd});const d=await r.json();
    document.getElementById('faceStatus').textContent=d.ok?'Learned: '+name+' (from file)':'Error: '+(d.error||'failed');
    document.getElementById('faceName').value='';document.getElementById('faceFile').value='';}catch(e){document.getElementById('faceStatus').textContent='Error: '+e.message;}
}
// v9.7.6 — auto-collect toggle + live status poll.
//
// Replaces the legacy calStart() / calStop() pair. Pair collection is
// automatic at server boot (DODO_AUTOCALIBRATE=1 by default in
// calibration.py); this button just lets you PAUSE if you're about to
// do something that would inject noise (move around, take the
// oximeter off, etc).
//
// The status poller hits /calibrate/status once per second and
// updates:
//   - The button label + colour (COLLECTING / PAUSED)
//   - A live n=<rppg>/<radar> counter so you can watch pairs
//     accumulate in real time
let _collectingClient = true;
async function toggleCollect(){
    const target = !_collectingClient;
    const url = target ? '/calibrate/resume' : '/calibrate/pause';
    try {
        const r = await fetch(url, {method: 'POST'});
        const d = await r.json();
        _collectingClient = !!d.collecting;
        _renderCollectButton();
    } catch(e){
        const el = document.getElementById('calStatus');
        if(el){ el.textContent = 'toggle error: ' + e.message; el.style.color = '#ef4444'; }
    }
}
function _renderCollectButton(){
    const btn = document.getElementById('collectToggle');
    if(!btn) return;
    if(_collectingClient){
        btn.textContent = 'COLLECTING ●';
        btn.style.background = '#22c55e';
        btn.style.color = '#000';
        btn.style.borderColor = '#16a34a';
        btn.title = 'Click to PAUSE pair collection (e.g. before deliberate motion)';
    } else {
        btn.textContent = 'PAUSED ⏸';
        btn.style.background = '#71717a';
        btn.style.color = '#fff';
        btn.style.borderColor = '#52525b';
        btn.title = 'Click to RESUME automatic pair collection';
    }
}
async function _refreshCalibrateStatus(){
    try {
        const r = await fetch('/calibrate/status');
        if(!r.ok) return;
        const d = await r.json();
        _collectingClient = !!d.collecting;
        _renderCollectButton();
        const el = document.getElementById('calStatus');
        if(el){
            el.textContent = 'pairs n=' + (d.n_rppg||0) + '/r=' + (d.n_radar||0);
            el.style.color = d.collecting ? '#22c55e' : '#71717a';
        }
    } catch(e){ /* silent — keep last good display */ }
}
// Poll once at load + every second.
document.addEventListener('DOMContentLoaded', function(){
    _refreshCalibrateStatus();
    setInterval(_refreshCalibrateStatus, 1000);
});

// v9.7.6 — train from accumulated pairs files. Unlike CAL rPPG / STOP
// CAL which only fit on samples collected during one button-press
// session, these endpoints reload the full calibration/{rppg,radar}_
// pairs.json from disk and run quality-gated regression (R² gate +
// outlier rejection per recompute_from_disk in calibration.py). They
// surface a structured result so you can see exactly what fit kind
// you got: "full" (good slope+intercept), "mean_offset" (R² too low,
// only the bias is meaningful), or "identity" (oxi was too constant
// to learn anything). Result colour: green=full, amber=mean_offset,
// red=identity or error.
function _renderTrainResult(d){
    const el = document.getElementById('trainResult');
    if(!el) return;
    if(!d || d.status === 'error' || d.status === undefined && !d.kind){
        el.textContent = '✗ ' + (d?.msg || 'training failed');
        el.style.color = '#ef4444';
        return;
    }
    let col = '#71717a';
    if(d.kind === 'full')        col = '#22c55e';
    else if(d.kind === 'mean_offset') col = '#f59e0b';
    else if(d.kind === 'identity')    col = '#ef4444';
    const label = d.which ? ('Radar ' + d.which.toUpperCase()) : 'rPPG';
    el.textContent = '✓ ' + label
        + ': ' + (d.kind || '?')
        + '  n=' + (d.n_used || 0) + '/' + (d.n_total || 0)
        + '  slope=' + (d.slope ?? 1).toFixed(3)
        + '  intercept=' + (d.intercept ?? 0).toFixed(2)
        + '  R²=' + (d.r_squared ?? 0).toFixed(3)
        + '  MAE=' + (d.mae_bpm ?? 0).toFixed(2);
    el.style.color = col;
    el.title = d.msg || '';
}
async function trainRppg(){
    const el = document.getElementById('trainResult');
    if(el){ el.textContent = 'Training rPPG…'; el.style.color = '#a1a1aa'; }
    try {
        const r = await fetch('/calibrate/recompute', {method: 'POST'});
        const d = await r.json();
        _renderTrainResult(d);
    } catch(e){
        if(el){ el.textContent = '✗ rPPG train: ' + e.message; el.style.color = '#ef4444'; }
    }
}
async function trainRadar(which){
    const el = document.getElementById('trainResult');
    if(el){ el.textContent = 'Training Radar ' + which.toUpperCase() + '…'; el.style.color = '#a1a1aa'; }
    try {
        const fd = new FormData(); fd.append('which', which);
        const r = await fetch('/calibrate/radar/recompute', {method: 'POST', body: fd});
        const d = await r.json();
        _renderTrainResult(d);
    } catch(e){
        if(el){ el.textContent = '✗ Radar ' + which.toUpperCase() + ' train: ' + e.message; el.style.color = '#ef4444'; }
    }
}
// ══════════════════════════════════════════
//  v8.7 Step 8 — 90-second emotional calibration lap
// ══════════════════════════════════════════
async function lapStart(){
    const subj=(document.getElementById('lapSubject')?.value||'').trim();
    const fd=new FormData();fd.append('subject',subj);fd.append('target_track','-1');
    try{
        const r=await fetch('/calibration/lap/start',{method:'POST',body:fd});
        const d=await r.json();
        if(d.error){alert('Lap error: '+d.error);return;}
        renderLapHud(d);
    }catch(e){alert('Lap start failed: '+e.message);}
}
async function lapAbort(){
    try{await fetch('/calibration/lap/abort',{method:'POST'});}catch(e){}
    const hud=document.getElementById('lapHud');if(hud)hud.style.display='none';
}
async function refreshLapProfiles(){
    try{
        const r=await fetch('/calibration/lap/profiles');const d=await r.json();
        const sel=document.getElementById('lapProfileSel');if(!sel)return;
        const cur=sel.value;
        sel.innerHTML='<option value="">— select profile —</option>'+
            (d.profiles||[]).map(p=>'<option value="'+p.filename+'">'+(p.subject||p.filename)+' ('+(p.duration_s||0).toFixed(0)+'s)</option>').join('');
        if(cur)sel.value=cur;
    }catch(e){}
}
async function lapLoadProfile(){
    const sel=document.getElementById('lapProfileSel');
    if(!sel||!sel.value){alert('Pick a profile first.');return;}
    const fd=new FormData();fd.append('filename',sel.value);fd.append('target_track','-1');
    try{
        const r=await fetch('/calibration/lap/load',{method:'POST',body:fd});
        const d=await r.json();
        if(d.ok){alert('Profile loaded onto track '+d.track_id+(d.subject?(' ('+d.subject+')'):''));}
        else{alert('Load failed: '+(d.error||'unknown'));}
    }catch(e){alert('Load failed: '+e.message);}
}
function renderLapHud(s){
    if(!s||(!s.active&&!s.completed&&!s.aborted)){
        const hud=document.getElementById('lapHud');if(hud)hud.style.display='none';return;
    }
    const hud=document.getElementById('lapHud');if(!hud)return;
    hud.style.display='block';
    if(s.completed){
        hud.style.borderColor='#22c55e';
        sEl('lapStage','LAP COMPLETE','#22c55e');
        sEl('lapTimer','done');
        document.getElementById('lapPrompt').textContent='Profile saved: '+(s.saved_path||'(see calibration/laps/)');
        document.getElementById('lapBar').style.width='100%';
        const sps=s.samples_per_stage||{};
        document.getElementById('lapSamples').textContent='samples — neutral:'+(sps.neutral||0)+' positive:'+(sps.positive||0)+' stress:'+(sps.stress||0);
        refreshLapProfiles();
        setTimeout(()=>{const h=document.getElementById('lapHud');if(h)h.style.display='none';},6000);
        return;
    }
    if(s.aborted){
        hud.style.borderColor='#ef4444';
        sEl('lapStage','LAP ABORTED','#ef4444');
        sEl('lapTimer','—');
        document.getElementById('lapPrompt').textContent='Calibration aborted. Partial profile preserved.';
        return;
    }
    hud.style.borderColor='#06b6d4';
    sEl('lapStage','STAGE '+(s.stage_index+1)+'/'+s.stage_total+' '+(s.stage||'').toUpperCase(),'#06b6d4');
    sEl('lapTimer',Math.ceil(s.stage_remaining_s)+'s');
    document.getElementById('lapPrompt').textContent=s.stage_prompt||'';
    const totalDur=s.total_duration_s||90;
    const elapsed=s.total_elapsed_s||0;
    const pct=Math.min(100,Math.max(0,(elapsed/totalDur)*100));
    document.getElementById('lapBar').style.width=pct+'%';
    const sps=s.samples_per_stage||{};
    document.getElementById('lapSamples').textContent='samples — neutral:'+(sps.neutral||0)+' positive:'+(sps.positive||0)+' stress:'+(sps.stress||0);
}
// Refresh profile list at boot + every 30s while idle
refreshLapProfiles();
setInterval(refreshLapProfiles,30000);
// Poll training status every 10s
setInterval(async()=>{
    try{const r=await fetch('/training/status');const d=await r.json();
    document.getElementById('trainCount').textContent='LLM:'+d.llm_pairs+' | rPPG:'+d.rppg_pairs+' | Faces:'+d.faces;}catch(e){}
},10000);
// v8.2: Poll governor stats every 30s for token budget gauge
setInterval(async()=>{
    try{const r=await fetch('/status');const s=await r.json();uGovBudget(s);}catch(e){}
},30000);

let inflight=0;const MAX_INFLIGHT=2;  // v9.1.5: hard cap latency at 2× server frame time
let ws=null;let wsReady=false;let wsQueue=0;let _wsDropCount=0;
// v8.4: adaptive JPEG quality — auto-tune based on round-trip latency
let _jpegQ=0.45;const _rttBuf=[];const _dfConfBuf=[];
// v9.7.4: optional URL-param overrides for the JPEG quality + mesh
// render cadence. Lets you trade visual fidelity for fps without
// editing code. Examples:
//   ?jpegq=0.30   — pin JPEG quality at 30%, faster wire payload
//   ?meshskip=2   — render landmark mesh on every 2nd frame only
//   ?meshskip=0   — never render the mesh (PAD ring + iris stay)
// URL params win over the adaptive controller, so power users can
// lock the system into a known state for benchmarking.
const _urlParams = new URLSearchParams(location.search || '');
const _jpegQOverride = parseFloat(_urlParams.get('jpegq') || '');
if(!isNaN(_jpegQOverride) && _jpegQOverride > 0 && _jpegQOverride <= 1){
    _jpegQ = _jpegQOverride;
}
let _meshRenderEvery = parseInt(_urlParams.get('meshskip') || '1', 10);
if(!Number.isFinite(_meshRenderEvery) || _meshRenderEvery < 0){
    _meshRenderEvery = 1;
}
let _meshRenderTick = 0;
// v9.7.1 — adaptive JPEG quality. Two signals drive the quality curve:
//   1. Round-trip latency (RTT)        — high RTT means we're either
//      bandwidth-starved or the server is overloaded; downshift quality
//      to shorten the wire payload.
//   2. DeepFace confidence (rolling)    — when DF confidence drops below
//      ~0.5, the server is having trouble extracting demographics from
//      the current pixels; bias upward by 0.05 so DF gets cleaner input.
// Bounds widened from [0.35, 0.55] to [0.30, 0.65] so we have headroom
// to bias up when DF needs better pixels and bias down when bandwidth
// is the bottleneck. Mode-1/full frames still get +0.15 in the caller
// (capped at 0.80 in v9.7.1, was 0.70).
function _adaptJpegQ(ms){
    // v9.7.4: when the user has pinned `?jpegq=...` in the URL we
    // disable the adaptive controller entirely — they want a fixed
    // quality for benchmarking or a predictable bandwidth budget.
    if(!isNaN(_jpegQOverride) && _jpegQOverride > 0 && _jpegQOverride <= 1) return;
    _rttBuf.push(ms);
    if(_rttBuf.length>5)_rttBuf.shift();
    if(_rttBuf.length<5) return;
    const avg=_rttBuf.reduce((a,b)=>a+b,0)/_rttBuf.length;
    // Bandwidth/server-load axis.
    if(avg>200) _jpegQ=Math.max(0.30,_jpegQ-0.02);
    else if(avg<80) _jpegQ=Math.min(0.65,_jpegQ+0.02);
    // DF-confidence axis (only adjust if we have a reading).
    if(_dfConfBuf.length>=5){
        const c=_dfConfBuf.reduce((a,b)=>a+b,0)/_dfConfBuf.length;
        if(c<0.4) _jpegQ=Math.min(0.65,_jpegQ+0.01);
        else if(c>0.8 && avg<150) _jpegQ=Math.max(0.30,_jpegQ-0.005);
    }
}
function _noteDfConf(c){
    if(typeof c!=='number'||!isFinite(c)) return;
    _dfConfBuf.push(c);
    if(_dfConfBuf.length>10) _dfConfBuf.shift();
}
// v8.4: session timer + frame counter
let _sessionStart=0;let _totalFrames=0;
function _updateSessionTimer(){if(!_sessionStart)return;const el=Math.floor((Date.now()-_sessionStart)/1000);const hh=String(Math.floor(el/3600)).padStart(2,'0');const mm=String(Math.floor((el%3600)/60)).padStart(2,'0');const ss=String(el%60).padStart(2,'0');sEl('sSession',hh+':'+mm+':'+ss+' | '+_totalFrames+' frames');}
// v8.4: hidden tab — drop to 1 FPS to save CPU/bandwidth
let _lastHiddenSend=0;

// ═══════════════════════════════════════════════════════════════════
// v9.3 plan steps 1, 2, 3, 4, 6, 8, 10 — client-side bootstrap
// ═══════════════════════════════════════════════════════════════════
//
// Feature flag resolution. The /webcam page embeds a <script id="ff-v93"
// type="application/json"> block with the server-resolved flag set.
// We parse it once and expose as window.FF.
const FF = (function(){
    try {
        const el = document.getElementById('ff-v93');
        if(!el) return {};
        const obj = JSON.parse(el.textContent || '{}');
        // ?safe=1 in the current URL forces every flag off (mirrors server).
        const qs = new URLSearchParams(location.search);
        if(qs.get('safe') === '1'){
            for(const k in obj) obj[k] = false;
        }
        // v9.5.x — URL-level per-flag override so operators can toggle a flag
        // without restarting the server. Example:
        //   http://localhost:8000/webcam?CLIENT_TRACKER=1
        // gives instant on-device tracking (zero WS round-trip for mesh).
        for(const k of Object.keys(obj)){
            const v = qs.get(k);
            if(v === '1' || v === 'true') obj[k] = true;
            else if(v === '0' || v === 'false') obj[k] = false;
        }
        return obj;
    } catch(e){ console.warn('[v9.3] FF parse failed', e); return {}; }
})();
window.FF = FF;
console.log('[v9.3] feature flags:', FF);

// ── v9.4.1 step 6 — ?pad_debug=1 operator view ──
// A floating panel listing every branch score per person (name,
// p_attack, reliability, latency_ms, meta dict). Reads from
// /pad/{pid} at 1 Hz. Keyboard-operable; `role="grid"` + aria-labels
// on each row give screen-reader support. Defaults off.
const PAD_DEBUG = (function(){
    try { return (new URLSearchParams(location.search)).get('pad_debug') === '1'; }
    catch(e){ return false; }
})();
function _padDebugMount(){
    if(!PAD_DEBUG || document.getElementById('padDebugPanel')) return;
    // v9.7.5: rewrite — draggable, collapsible, position-persistent.
    // The previous bottom-right fixed position permanently overlapped
    // the Dodo Intelligence card and made both unreadable. New panel:
    //   - Header with drag handle, ─ (collapse) and × (hide) buttons
    //   - Dragging persists position to localStorage
    //   - Collapse state persists too
    //   - When hidden, a small `[PAD]` chip in the corner can re-open
    //   - ESC hides; Ctrl+Shift+P toggles
    const p = document.createElement('div');
    p.id = 'padDebugPanel';
    p.setAttribute('role', 'region');
    p.setAttribute('aria-label', 'PAD operator debug view');
    p.tabIndex = 0;
    // Read persisted position/collapse state.
    let _saved = {};
    try { _saved = JSON.parse(localStorage.getItem('padDebugState')||'{}') || {}; }
    catch(e){}
    const _left = _saved.left;
    const _top  = _saved.top;
    const _collapsed = !!_saved.collapsed;
    // If a saved position exists use it; otherwise default to top-right
    // (out of the way of the bottom-center Dodo Intelligence card).
    const posStyle = (typeof _left === 'number' && typeof _top === 'number')
        ? 'left:'+_left+'px;top:'+_top+'px;'
        : 'right:8px;top:120px;';
    p.style.cssText = 'position:fixed;'+posStyle+'z-index:9999;'
        + 'width:360px;max-height:60vh;overflow:hidden;'
        + 'background:#111;color:#fafafa;border:1px solid #a855f7;'
        + 'border-radius:6px;font-family:monospace;font-size:11px;'
        + 'box-shadow:0 0 12px rgba(168,85,247,.35);'
        + 'display:flex;flex-direction:column';
    p.innerHTML =
        '<div id="padDebugHeader" style="cursor:move;user-select:none;'
            + 'display:flex;align-items:center;gap:6px;'
            + 'padding:6px 8px;background:#1a1a1a;border-bottom:1px solid #27272a">'
            + '<span style="color:#a855f7;font-weight:800;font-size:11px;'
                + 'letter-spacing:1px;text-transform:uppercase">PAD DEBUG</span>'
            + '<span style="flex:1"></span>'
            + '<button id="padDebugCollapse" title="Collapse / expand body" '
                + 'style="background:transparent;border:1px solid #3f3f46;'
                + 'color:#a855f7;cursor:pointer;width:22px;height:18px;'
                + 'border-radius:3px;font-size:11px;line-height:1;'
                + 'padding:0">'+(_collapsed?'+':'–')+'</button>'
            + '<button id="padDebugClose" title="Hide panel (re-open via [PAD] chip or Ctrl+Shift+P)" '
                + 'style="background:transparent;border:1px solid #3f3f46;'
                + 'color:#ef4444;cursor:pointer;width:22px;height:18px;'
                + 'border-radius:3px;font-size:13px;line-height:1;'
                + 'padding:0">×</button>'
        + '</div>'
        + '<div id="padDebugContent" style="overflow-y:auto;padding:8px;'
            + (_collapsed?'display:none':'')+'">'
            + '<div id="padDebugBody" role="grid" aria-label="PAD branch scores"></div>'
        + '</div>';
    document.body.appendChild(p);

    // ── Tiny [PAD] re-open chip, hidden until user closes the panel ──
    const chip = document.createElement('div');
    chip.id = 'padDebugChip';
    chip.title = 'Re-open PAD DEBUG panel';
    chip.textContent = 'PAD';
    chip.style.cssText = 'position:fixed;right:8px;bottom:8px;z-index:9999;'
        + 'background:#111;color:#a855f7;border:1px solid #a855f7;'
        + 'border-radius:4px;padding:4px 8px;font-family:monospace;'
        + 'font-size:10px;font-weight:800;cursor:pointer;display:none;'
        + 'box-shadow:0 0 6px rgba(168,85,247,.35)';
    document.body.appendChild(chip);

    // ── Drag from header ──
    const header = p.querySelector('#padDebugHeader');
    let _dragging = false, _dragDX = 0, _dragDY = 0;
    header.addEventListener('mousedown', function(ev){
        if(ev.target.tagName === 'BUTTON') return;  // buttons don't drag
        _dragging = true;
        const r = p.getBoundingClientRect();
        _dragDX = ev.clientX - r.left;
        _dragDY = ev.clientY - r.top;
        ev.preventDefault();
    });
    document.addEventListener('mousemove', function(ev){
        if(!_dragging) return;
        const newLeft = Math.max(0, Math.min(window.innerWidth - 80,  ev.clientX - _dragDX));
        const newTop  = Math.max(0, Math.min(window.innerHeight - 32, ev.clientY - _dragDY));
        p.style.left  = newLeft + 'px';
        p.style.top   = newTop  + 'px';
        p.style.right = 'auto';
        p.style.bottom = 'auto';
    });
    document.addEventListener('mouseup', function(){
        if(!_dragging) return;
        _dragging = false;
        // Persist
        try {
            const r = p.getBoundingClientRect();
            const cur = JSON.parse(localStorage.getItem('padDebugState')||'{}');
            cur.left = Math.round(r.left);
            cur.top  = Math.round(r.top);
            localStorage.setItem('padDebugState', JSON.stringify(cur));
        } catch(e){}
    });

    // ── Collapse / expand ──
    const collapseBtn = p.querySelector('#padDebugCollapse');
    const content = p.querySelector('#padDebugContent');
    collapseBtn.addEventListener('click', function(){
        const isCollapsed = content.style.display === 'none';
        content.style.display = isCollapsed ? '' : 'none';
        collapseBtn.textContent = isCollapsed ? '–' : '+';
        try {
            const cur = JSON.parse(localStorage.getItem('padDebugState')||'{}');
            cur.collapsed = !isCollapsed;
            localStorage.setItem('padDebugState', JSON.stringify(cur));
        } catch(e){}
    });

    // ── Close (hide) + re-open via chip ──
    const closeBtn = p.querySelector('#padDebugClose');
    function _hidePanel(){
        p.style.display = 'none';
        chip.style.display = 'block';
        try {
            const cur = JSON.parse(localStorage.getItem('padDebugState')||'{}');
            cur.hidden = true;
            localStorage.setItem('padDebugState', JSON.stringify(cur));
        } catch(e){}
    }
    function _showPanel(){
        p.style.display = 'flex';
        chip.style.display = 'none';
        try {
            const cur = JSON.parse(localStorage.getItem('padDebugState')||'{}');
            cur.hidden = false;
            localStorage.setItem('padDebugState', JSON.stringify(cur));
        } catch(e){}
    }
    closeBtn.addEventListener('click', _hidePanel);
    chip.addEventListener('click', _showPanel);
    if(_saved.hidden){ _hidePanel(); }

    // ── Keyboard: ESC hides, Ctrl+Shift+P toggles ──
    p.addEventListener('keydown', function(ev){
        if(ev.key === 'Escape'){ _hidePanel(); }
    });
    document.addEventListener('keydown', function(ev){
        if(ev.ctrlKey && ev.shiftKey && (ev.key === 'P' || ev.key === 'p')){
            (p.style.display === 'none') ? _showPanel() : _hidePanel();
            ev.preventDefault();
        }
    });

    // v9.8.3 — screen-flash active liveness (delegated; buttons re-created each poll).
    const padBody = p.querySelector('#padDebugBody');
    if(padBody){
        padBody.addEventListener('click', async function(ev){
            const btn = ev.target.closest('[data-pad-flash]');
            if(!btn) return;
            ev.preventDefault();
            const flashPid = btn.getAttribute('data-pad-flash');
            btn.disabled = true;
            try {
                const r = await fetch('/pad/challenge/screen_flash/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({pid: parseInt(flashPid, 10), duration_ms: 140}),
                });
                const j = await r.json();
                if(j.error){ btn.textContent = 'err'; return; }
                const hex = j.color_hex || '#ffffff';
                const ms = j.duration_ms || 120;
                const prev = document.body.style.backgroundColor;
                document.body.style.backgroundColor = hex;
                setTimeout(function(){ document.body.style.backgroundColor = prev; }, ms);
                btn.textContent = (j.color_name || 'ok').slice(0, 6);
            } catch(e){
                btn.textContent = 'fail';
            } finally {
                setTimeout(function(){ btn.disabled = false; btn.textContent = 'flash'; }, 800);
            }
        });
    }
}
// v9.7.5: PAIRS badge poll. Hits /qlora/status once per second and
// updates the top-of-page pair count. Color shifts from grey when
// far below the 1000-pair export threshold → amber as we approach
// → green once eligible. Tooltip shows total + this-session split.
async function _qloraBadgeRefresh(){
    const el = document.getElementById('pairBadge');
    if(!el) return;
    try {
        const r = await fetch('/qlora/status');
        if(!r.ok) return;
        const d = await r.json();
        const total   = d.total || 0;
        const session = d.this_session || 0;
        const need    = d.min_for_export || 1000;
        // Color tier
        let col = '#a855f7';   // default purple
        let bg  = '#18181b';
        if(total >= need){          col = '#22c55e'; bg = '#0a1f0a'; }   // green: ready
        else if(total >= need*0.5){ col = '#f59e0b'; bg = '#1f1505'; }   // amber: halfway
        else {                       col = '#a855f7'; bg = '#18181b'; }  // purple: warming
        el.textContent = 'PAIRS: ' + total;
        el.style.color = col;
        el.style.background = bg;
        el.style.borderColor = col;
        el.title = 'QLoRA training pairs\n'
                 + '  total on disk: ' + total + '\n'
                 + '  this session:  ' + session + '\n'
                 + '  min for export: ' + need
                 + (total >= need ? ' ✓ READY' : '  (need ' + (need - total) + ' more)');
    } catch(e){
        // Silent — keep last good value if endpoint hiccups
    }
}
// Poll on page load + every 1 s thereafter.
document.addEventListener('DOMContentLoaded', function(){
    _qloraBadgeRefresh();
    setInterval(_qloraBadgeRefresh, 1000);
});


async function _padDebugRefresh(){
    if(!PAD_DEBUG) return;
    const body = document.getElementById('padDebugBody');
    if(!body) return;
    const pids = [];
    try {
        for(const f of (window._v93 && window._v93.instant && window._v93.instant.faces) || []){
            // v9.8.1: prefer server-assigned f.pid (track_id) when present,
            // fall back to f.i (slot index, always 0..N-1) for client-only
            // tracker output. Same change applied to _padAgreeRefresh below.
            if(f && (f.pid !== undefined || f.i !== undefined)) pids.push(f.pid !== undefined ? f.pid : f.i);
        }
    } catch(e){}
    if(pids.length === 0){ body.innerHTML = '<div style="color:#52525b">(no tracked persons)</div>'; return; }
    const rows = [];
    for(const pid of pids){
        try {
            const r = await fetch('/pad/' + pid);
            const j = await r.json();
            if(j && j.last){
                const vCol = j.last.verdict === 'SPOOF' ? '#ef4444'
                    : j.last.verdict === 'UNCERTAIN' ? '#f59e0b' : '#22c55e';
                let rowHtml = '<div role="rowgroup" aria-label="pid '+pid+'" style="border-top:1px solid #27272a;padding:3px 0">';
                rowHtml += '<div style="color:'+vCol+';font-weight:800">pid='+pid
                    +' · '+j.last.verdict
                    +' · p='+(j.last.p_attack||0).toFixed(3)
                    +' · class='+(j.last.attack_class||'?')
                    +' · log_odds='+(j.last.log_odds||0).toFixed(2)
                    +' <button type="button" data-pad-flash="'+pid+'" title="Active liveness: pulse background + score skin reflection" '
                    +'style="margin-left:6px;background:#27272a;border:1px solid #a855f7;color:#a855f7;'
                    +'cursor:pointer;font-size:9px;padding:1px 5px;border-radius:3px">flash</button></div>';
                const ins = j.insights || {};
                if(ins.abstentions && ins.abstentions.length){
                    rowHtml += '<div style="padding-left:8px;color:#71717a;font-size:9px">abstain: '
                        + ins.abstentions.map(function(a){ return a.branch+':'+a.reason; }).join(' · ')
                        + '</div>';
                }
                if(ins.fusion_meta_subset && Object.keys(ins.fusion_meta_subset).length){
                    rowHtml += '<div style="padding-left:8px;color:#52525b;font-size:9px">fusion: '
                        + JSON.stringify(ins.fusion_meta_subset) + '</div>';
                }
                for(const b of (j.last.branches||[])){
                    const metaStr = JSON.stringify(b.meta || {});
                    // v9.8: inline annotations for the EMA-baseline state
                    // and auto-disable state (mostly for patch_depth, but
                    // any branch can opt in by setting these meta keys).
                    const m = b.meta || {};
                    let tags = '';
                    if(m.disabled_for_session){
                        tags += ' [<span style="color:#71717a;background:#27272a;padding:0 4px;border-radius:2px">DISABLED</span>]';
                    } else if(typeof m.secs_high === 'number' && m.secs_high > 10){
                        tags += ' [<span style="color:#f59e0b">stuck '+m.secs_high.toFixed(0)+'s</span>]';
                    }
                    if(typeof m.subject_baseline === 'number' && typeof m.raw_jump === 'number'){
                        const _j = m.raw_jump;
                        const _jcol = _j > 0.20 ? '#ef4444' : _j > 0.05 ? '#f59e0b' : '#52525b';
                        tags += ' base='+m.subject_baseline.toFixed(2)
                              + ' <span style="color:'+_jcol+'">jump='+(_j>=0?'+':'')+_j.toFixed(2)+'</span>';
                    }
                    if(typeof m.saturated_run === 'number' && m.saturated_run >= 1){
                        tags += ' [<span style="color:#ef4444">SAT</span>]';
                    }
                    rowHtml += '<div role="row" aria-label="branch '+b.name+'" '
                        + 'style="padding-left:8px;color:#a1a1aa">'
                        + '<span style="color:#fafafa">'+b.name+'</span>'
                        + ' p='+(b.p_attack||0).toFixed(3)
                        + ' rel='+(b.reliability||0).toFixed(2)
                        + ' lat='+(b.latency_ms||0).toFixed(1)+'ms'
                        + (b.dominant_class ? ' [<span style="color:#ef4444">'+b.dominant_class+'</span>]' : '')
                        + tags
                        + '<div style="color:#52525b;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title='+JSON.stringify(metaStr)+'>'+metaStr+'</div></div>';
                }
                rowHtml += '</div>';
                rows.push(rowHtml);
            } else if (j && j.__umn_wiring_bug__) {
                // v9.7.0-merged 2026-05-11: explicit marker that the
                // server returned a NameError/AttributeError (modular-
                // refactor handler broken), not just "no data yet".
                // Render a visible WIRING-BUG badge so the operator can
                // tell us instead of waiting silently.
                const errShort = (j.error || 'unknown').substring(0, 80);
                rows.push('<div style="color:#fafafa;background:#7f1d1d;padding:4px 6px;border-radius:3px">'
                    + 'pid=' + pid + ' · <b>WIRING BUG</b> · ' + errShort + '</div>');
            } else if (j && j.n_persons === 0){
                // v9.8.1: same hint as PAD AGREEMENT — surface why pid=0
                // returned empty (no server-side track for any face yet).
                rows.push('<div style="color:#52525b">pid='+pid+' — server has no tracked persons</div>');
            } else {
                rows.push('<div style="color:#52525b">pid='+pid+' — warming (no fused PAD yet)</div>');
            }
        } catch(e){ rows.push('<div style="color:#ef4444">pid='+pid+' — fetch err</div>'); }
    }
    body.innerHTML = rows.join('');
}
if(PAD_DEBUG){
    document.addEventListener('DOMContentLoaded', function(){
        _padDebugMount();
        setInterval(_padDebugRefresh, 1000);
    });
}

// ══════════════════════════════════════════════════════════════════
//  v9.7.6 — PAD AGREEMENT panel
// ══════════════════════════════════════════════════════════════════
// Companion to PAD DEBUG. PAD DEBUG dumps raw per-branch numbers; this
// panel synthesises them into an agreement view answering "do the
// branches agree on LIVE vs SPOOF, or are they fighting?".
//
// Per-branch vote rule (mirrors main.py:_pad_run_once `DISAGREE_REL_MIN`
// + p-margin gate at line 4429): a branch is COUNTED only if
//   reliability >= 0.25  AND  |p_attack - 0.5| > 0.20
// Otherwise the branch is in warmup / no-opinion and we don't include
// it in the agreement math (rendered separately as "abstaining").
//
// Vote = LIVE if p_attack < 0.30; SPOOF if p_attack > 0.70; otherwise
// LEAN_LIVE / LEAN_SPOOF / UNCERTAIN (the bands match the `attack_class`
// emission thresholds in pad/types.py).
//
// Agreement metric = (# branches voting same as fused verdict) /
//                    (# branches with a vote at all)
// 100% = all confident branches agree with the fusion. <50% = half
// of them are voting against the fused verdict — fusion is overruling
// a strong minority and the operator should look at WHY.
//
// Disagreement spread = max(p_attack) - min(p_attack) among confident
// branches. <0.20 = tight cluster (high confidence). >0.50 = wide
// scatter (branches see fundamentally different evidence).
//
// Activation: ?pad_agree=1 (independent of pad_debug). Both can be on.
const PAD_AGREE = (function(){
    try { return (new URLSearchParams(location.search)).get('pad_agree') === '1'; }
    catch(e){ return false; }
})();
function _padAgreeMount(){
    if(!PAD_AGREE || document.getElementById('padAgreePanel')) return;
    const p = document.createElement('div');
    p.id = 'padAgreePanel';
    p.setAttribute('role', 'region');
    p.setAttribute('aria-label', 'PAD branch agreement view');
    p.tabIndex = 0;
    let _saved = {};
    try { _saved = JSON.parse(localStorage.getItem('padAgreeState')||'{}') || {}; }
    catch(e){}
    const _left = _saved.left;
    const _top  = _saved.top;
    const _collapsed = !!_saved.collapsed;
    // Default to top-LEFT so we don't overlap PAD DEBUG (top-right).
    const posStyle = (typeof _left === 'number' && typeof _top === 'number')
        ? 'left:'+_left+'px;top:'+_top+'px;'
        : 'left:8px;top:120px;';
    p.style.cssText = 'position:fixed;'+posStyle+'z-index:9999;'
        + 'width:360px;max-height:60vh;overflow:hidden;'
        + 'background:#111;color:#fafafa;border:1px solid #22d3ee;'
        + 'border-radius:6px;font-family:monospace;font-size:11px;'
        + 'box-shadow:0 0 12px rgba(34,211,238,.35);'
        + 'display:flex;flex-direction:column';
    p.innerHTML =
        '<div id="padAgreeHeader" style="cursor:move;user-select:none;'
            + 'display:flex;align-items:center;gap:6px;'
            + 'padding:6px 8px;background:#1a1a1a;border-bottom:1px solid #27272a">'
            + '<span style="color:#22d3ee;font-weight:800;font-size:11px;'
                + 'letter-spacing:1px;text-transform:uppercase">PAD AGREEMENT</span>'
            + '<span style="flex:1"></span>'
            + '<button id="padAgreeCollapse" title="Collapse / expand body" '
                + 'style="background:transparent;border:1px solid #3f3f46;'
                + 'color:#22d3ee;cursor:pointer;width:22px;height:18px;'
                + 'border-radius:3px;font-size:11px;line-height:1;'
                + 'padding:0">'+(_collapsed?'+':'–')+'</button>'
            + '<button id="padAgreeClose" title="Hide panel (re-open via [PAD-A] chip or Ctrl+Shift+A)" '
                + 'style="background:transparent;border:1px solid #3f3f46;'
                + 'color:#ef4444;cursor:pointer;width:22px;height:18px;'
                + 'border-radius:3px;font-size:13px;line-height:1;'
                + 'padding:0">×</button>'
        + '</div>'
        + '<div id="padAgreeContent" style="overflow-y:auto;padding:8px;'
            + (_collapsed?'display:none':'')+'">'
            + '<div id="padAgreeHint" style="color:#52525b;font-size:9px;line-height:1.35;padding:0 0 6px;margin-bottom:4px;border-bottom:1px solid #27272a">'
            + 'Dynamics is per-face (v9.8.2). High-fidelity phone replay is intentionally weak for the dynamics branch — use moire, patch_depth, deepfake, rPPG, and radar. PAD flags: GET /health/full → subsystems.pad.'
            + '</div>'
            + '<div id="padAgreeBody" role="grid" aria-label="PAD branch agreement"></div>'
        + '</div>';
    document.body.appendChild(p);

    const chip = document.createElement('div');
    chip.id = 'padAgreeChip';
    chip.title = 'Re-open PAD AGREEMENT panel';
    chip.textContent = 'PAD-A';
    chip.style.cssText = 'position:fixed;left:8px;bottom:8px;z-index:9999;'
        + 'background:#111;color:#22d3ee;border:1px solid #22d3ee;'
        + 'border-radius:4px;padding:4px 8px;font-family:monospace;'
        + 'font-size:10px;font-weight:800;cursor:pointer;display:none;'
        + 'box-shadow:0 0 6px rgba(34,211,238,.35)';
    document.body.appendChild(chip);

    // Drag from header (mirror PAD DEBUG drag impl)
    const header = p.querySelector('#padAgreeHeader');
    let _dragging = false, _dragDX = 0, _dragDY = 0;
    header.addEventListener('mousedown', function(ev){
        if(ev.target.tagName === 'BUTTON') return;
        _dragging = true;
        const r = p.getBoundingClientRect();
        _dragDX = ev.clientX - r.left;
        _dragDY = ev.clientY - r.top;
        ev.preventDefault();
    });
    document.addEventListener('mousemove', function(ev){
        if(!_dragging) return;
        const newLeft = Math.max(0, Math.min(window.innerWidth - 80,  ev.clientX - _dragDX));
        const newTop  = Math.max(0, Math.min(window.innerHeight - 32, ev.clientY - _dragDY));
        p.style.left  = newLeft + 'px';
        p.style.top   = newTop  + 'px';
        p.style.right = 'auto';
        p.style.bottom = 'auto';
    });
    document.addEventListener('mouseup', function(){
        if(!_dragging) return;
        _dragging = false;
        try {
            const r = p.getBoundingClientRect();
            const cur = JSON.parse(localStorage.getItem('padAgreeState')||'{}');
            cur.left = Math.round(r.left);
            cur.top  = Math.round(r.top);
            localStorage.setItem('padAgreeState', JSON.stringify(cur));
        } catch(e){}
    });

    const collapseBtn = p.querySelector('#padAgreeCollapse');
    const content = p.querySelector('#padAgreeContent');
    collapseBtn.addEventListener('click', function(){
        const isCollapsed = content.style.display === 'none';
        content.style.display = isCollapsed ? '' : 'none';
        collapseBtn.textContent = isCollapsed ? '–' : '+';
        try {
            const cur = JSON.parse(localStorage.getItem('padAgreeState')||'{}');
            cur.collapsed = !isCollapsed;
            localStorage.setItem('padAgreeState', JSON.stringify(cur));
        } catch(e){}
    });

    const closeBtn = p.querySelector('#padAgreeClose');
    function _hidePanel(){
        p.style.display = 'none';
        chip.style.display = 'block';
        try {
            const cur = JSON.parse(localStorage.getItem('padAgreeState')||'{}');
            cur.hidden = true;
            localStorage.setItem('padAgreeState', JSON.stringify(cur));
        } catch(e){}
    }
    function _showPanel(){
        p.style.display = 'flex';
        chip.style.display = 'none';
        try {
            const cur = JSON.parse(localStorage.getItem('padAgreeState')||'{}');
            cur.hidden = false;
            localStorage.setItem('padAgreeState', JSON.stringify(cur));
        } catch(e){}
    }
    closeBtn.addEventListener('click', _hidePanel);
    chip.addEventListener('click', _showPanel);
    if(_saved.hidden){ _hidePanel(); }

    p.addEventListener('keydown', function(ev){
        if(ev.key === 'Escape'){ _hidePanel(); }
    });
    document.addEventListener('keydown', function(ev){
        if(ev.ctrlKey && ev.shiftKey && (ev.key === 'A' || ev.key === 'a')){
            (p.style.display === 'none') ? _showPanel() : _hidePanel();
            ev.preventDefault();
        }
    });
}

// Map a per-branch p_attack to a coarse vote label. The 0.30 / 0.70
// thresholds match the LIVE / SPOOF emission gates in pad/types.py;
// the middle band is split into LEAN_LIVE / LEAN_SPOOF so the user
// can distinguish "weakly says live" from "actively undecided".
function _padAgreeVote(p){
    if(p == null) return 'NA';
    if(p < 0.30) return 'LIVE';
    if(p < 0.45) return 'LEAN_LIVE';
    if(p > 0.70) return 'SPOOF';
    if(p > 0.55) return 'LEAN_SPOOF';
    return 'UNCERTAIN';
}
function _padAgreeVoteColor(v){
    if(v === 'LIVE')       return '#22c55e';
    if(v === 'LEAN_LIVE')  return '#84cc16';
    if(v === 'UNCERTAIN')  return '#f59e0b';
    if(v === 'LEAN_SPOOF') return '#fb923c';
    if(v === 'SPOOF')      return '#ef4444';
    return '#52525b';
}
// Two-direction comparison: "agrees with the fused verdict at the
// LIVE/SPOOF axis level". LEAN_* counts as agreeing on direction.
function _padAgreeOnAxis(branchVote, fusedVerdict){
    if(branchVote === 'NA' || branchVote === 'UNCERTAIN') return null;  // no opinion
    const isLive  = (branchVote === 'LIVE'  || branchVote === 'LEAN_LIVE');
    const isSpoof = (branchVote === 'SPOOF' || branchVote === 'LEAN_SPOOF');
    if(fusedVerdict === 'LIVE')  return isLive;
    if(fusedVerdict === 'SPOOF') return isSpoof;
    return null;  // fused is UNCERTAIN — nothing to agree with
}

async function _padAgreeRefresh(){
    if(!PAD_AGREE) return;
    const body = document.getElementById('padAgreeBody');
    if(!body) return;
    const pids = [];
    try {
        for(const f of (window._v93 && window._v93.instant && window._v93.instant.faces) || []){
            if(f && (f.pid !== undefined || f.i !== undefined)) pids.push(f.pid !== undefined ? f.pid : f.i);
        }
    } catch(e){}
    if(pids.length === 0){ body.innerHTML = '<div style="color:#52525b">(no tracked persons)</div>'; return; }
    const rows = [];
    const REL_MIN = 0.25;          // mirror DISAGREE_REL_MIN in main.py:4429
    const P_MARGIN = 0.20;         // mirror p-margin gate in same comment block
    for(const pid of pids){
        try {
            const r = await fetch('/pad/' + pid);
            const j = await r.json();
            if(!j || !j.last){
                // v9.8.1: differentiate "server has no tracked persons at all"
                // from "server has tracks but PAD branches haven't fused yet".
                // n_persons==0 means the WS ingest path has not yet bound any
                // face to a track_id → typically means the browser hasn't
                // pushed mode=1 frames yet, or the face detector skipped this
                // frame, or the server tracker is gated on confidence. Show
                // a useful hint instead of repeating "pid=0 — no data".
                if(j && j.n_persons === 0){
                    rows.push('<div style="color:#52525b">pid='+pid+' — server has no tracked persons (push mode=1 frames or hold still in frame)</div>');
                } else {
                    rows.push('<div style="color:#52525b">pid='+pid+' — warming (no fused PAD yet)</div>');
                }
                continue;
            }
            const last = j.last;
            const fusedVerdict = last.verdict || 'UNCERTAIN';
            const branches = last.branches || [];
            // v9.7.6 — fusion-level DEGRADED indicator. Set by
            // pad.fusion.fuse() when the verdict was downgraded from
            // LIVE → UNCERTAIN because no physical liveness sensor
            // (radar_presence with rel ≥ 0.30 or rppg with rel ≥ 0.30)
            // confirmed presence. Visual-only branches (patch_depth /
            // moire / deepfake) can be fooled by a high-quality phone
            // screen, so we refuse confident LIVE in their absence.
            const fusedMeta = last.meta || {};
            const degraded = !!fusedMeta.degraded_no_physical;

            // Score every branch. "Confident" = passes both reliability
            // AND p-margin gates — these are the branches that should
            // contribute to fusion.
            let confident = 0, agreeing = 0, opposing = 0, abstain = 0;
            let pMin =  1.0, pMax = 0.0;
            const branchRows = [];
            for(const b of branches){
                const p   = b.p_attack ?? 0.5;
                const rel = b.reliability ?? 0.0;
                const margin = Math.abs(p - 0.5);
                const isConfident = (rel >= REL_MIN) && (margin > P_MARGIN);
                const vote = _padAgreeVote(p);
                const voteCol = _padAgreeVoteColor(vote);
                let agreeFlag = '   ';
                let agreeCol = '#52525b';
                if(!isConfident){
                    abstain++;
                    agreeFlag = '···';
                    agreeCol = '#52525b';
                } else {
                    confident++;
                    pMin = Math.min(pMin, p);
                    pMax = Math.max(pMax, p);
                    const onAxis = _padAgreeOnAxis(vote, fusedVerdict);
                    if(onAxis === true){
                        agreeing++;
                        agreeFlag = ' ✓ ';
                        agreeCol = '#22c55e';
                    } else if(onAxis === false){
                        opposing++;
                        agreeFlag = ' ✗ ';
                        agreeCol = '#ef4444';
                    } else {
                        agreeFlag = ' ? ';
                        agreeCol = '#a1a1aa';
                    }
                }
                branchRows.push(
                    '<div role="row" aria-label="branch '+b.name+'" '
                        + 'style="padding-left:8px;display:grid;'
                        + 'grid-template-columns:18px 1fr 60px 64px 50px;'
                        + 'gap:4px;align-items:center">'
                        + '<span style="color:'+agreeCol+';font-weight:800">'+agreeFlag+'</span>'
                        + '<span style="color:#fafafa">'+b.name+'</span>'
                        + '<span style="color:'+voteCol+';font-size:9px;text-align:right">'+vote+'</span>'
                        + '<span style="color:#a1a1aa;text-align:right">p='+p.toFixed(3)+'</span>'
                        + '<span style="color:#71717a;text-align:right">r='+rel.toFixed(2)+'</span>'
                    + '</div>'
                );
            }

            const total = agreeing + opposing;
            const agreePct = total > 0 ? Math.round(100 * agreeing / total) : null;
            const spread = (confident > 0) ? (pMax - pMin) : 0;
            // Header colour by agreement strength
            let hdrCol = '#52525b';
            if(agreePct === null)        hdrCol = '#a1a1aa';   // no confident branches
            else if(agreePct >= 80)      hdrCol = '#22c55e';   // strong consensus
            else if(agreePct >= 60)      hdrCol = '#84cc16';   // majority
            else if(agreePct >= 40)      hdrCol = '#f59e0b';   // split
            else                         hdrCol = '#ef4444';   // minority overruled
            const fusedCol = fusedVerdict === 'SPOOF' ? '#ef4444'
                : fusedVerdict === 'UNCERTAIN' ? '#f59e0b' : '#22c55e';
            const agreeStr = (agreePct === null)
                ? '—  (no confident votes yet)'
                : agreePct + '%  (' + agreeing + '/' + total + ' agree, ' + opposing + ' oppose)';
            // v9.7.6 — DEGRADED banner when fusion downgraded due to
            // missing physical liveness signal (radar OFFLINE + no
            // rppg lock). Renders at the top of the per-pid block,
            // amber background with explanation.
            const degradedBanner = degraded
                ? '<div style="background:#451a03;color:#fbbf24;padding:4px 6px;'
                  + 'border:1px solid #f59e0b;border-radius:3px;'
                  + 'font-size:10px;margin:3px 0">'
                  + '⚠ DEGRADED — visual branches voted LIVE but radar + rppg are OFFLINE. '
                  + 'High-res phone screens can defeat visual-only PAD. Verdict downgraded to UNCERTAIN '
                  + 'until at least one physical sensor confirms presence.'
                  + '</div>'
                : '';
            let rowHtml = '<div role="rowgroup" aria-label="pid '+pid+'" style="border-top:1px solid #27272a;padding:4px 0">'
                + '<div style="font-weight:800;color:'+hdrCol+'">'
                    + 'pid='+pid
                    + ' · fused=<span style="color:'+fusedCol+'">'+fusedVerdict+'</span>'
                    + (degraded ? ' <span style="color:#f59e0b">⚠</span>' : '')
                + '</div>'
                + degradedBanner
                + '<div style="color:#a1a1aa;font-size:10px;padding:2px 0 4px">'
                    + 'agreement: <span style="color:'+hdrCol+';font-weight:800">'+agreeStr+'</span>'
                    + ' · abstaining: '+abstain
                    + ' · spread Δp: '+spread.toFixed(2)
                + '</div>'
                + branchRows.join('')
                + '</div>';
            rows.push(rowHtml);
        } catch(e){
            rows.push('<div style="color:#ef4444">pid='+pid+' — fetch err</div>');
        }
    }
    body.innerHTML = rows.join('');
}
if(PAD_AGREE){
    document.addEventListener('DOMContentLoaded', function(){
        _padAgreeMount();
        setInterval(_padAgreeRefresh, 1000);
    });
}

// ── v9.4.1 step 8 — PAD anomaly HUD banner ──
// Polls /pad/anomalies/recent every 10 s. Shows an amber banner when
// the 60-s disagreement rate exceeds the gate; red once an anomaly
// actually fires (sustained > 60 s). Gated by FF.PAD_UI.
function _padAnomalyMount(){
    if(document.getElementById('padAnomalyBanner')) return;
    const b = document.createElement('div');
    b.id = 'padAnomalyBanner';
    b.setAttribute('role', 'status');
    b.setAttribute('aria-live', 'polite');
    b.style.cssText = 'display:none;position:fixed;left:50%;top:6px;'
        + 'transform:translateX(-50%);z-index:9998;padding:6px 12px;'
        + 'border-radius:4px;font-family:monospace;font-size:11px;'
        + 'font-weight:700;letter-spacing:1px;text-transform:uppercase';
    document.body.appendChild(b);
}
async function _padAnomalyPoll(){
    if(!(FF && FF.PAD_UI)) return;
    const banner = document.getElementById('padAnomalyBanner');
    if(!banner) return;
    try {
        const r = await fetch('/pad/anomalies/recent');
        const j = await r.json();
        const rate = (j && typeof j.current_rate === 'number') ? j.current_rate : 0;
        const gate = (j && typeof j.gate === 'number') ? j.gate : 0.25;
        const hasAnomaly = j && Array.isArray(j.recent) && j.recent.length > 0;
        if(hasAnomaly){
            banner.style.display = 'block';
            banner.style.background = '#7f1d1d';
            banner.style.color = '#fef2f2';
            banner.textContent = 'PAD ANOMALY · disagreement ' + (rate*100).toFixed(0)
                + '% · ' + j.recent.length + ' sustained events';
        } else if(rate > gate){
            banner.style.display = 'block';
            banner.style.background = '#78350f';
            banner.style.color = '#fef3c7';
            banner.textContent = 'PAD WARN · disagreement ' + (rate*100).toFixed(0) + '%';
        } else {
            banner.style.display = 'none';
        }
    } catch(e){ /* quiet */ }
}
document.addEventListener('DOMContentLoaded', function(){
    _padAnomalyMount();
    setInterval(_padAnomalyPoll, 10000);
});

// Step 1 — INSTANT vs BEST-EFFORT tiers. Two independent data planes.
// The tracker/overlay read only `instant.*`; the WS onmessage writes
// only `server.*` signals. The two never share a render path.
const instant = {
    faces: [],            // [{pid, lmk, blendshapes, pose, gaze}]
    emotion: new Map(),   // pid -> {label, confidence, ts}
    frameTs: 0,
    // v9.4.1 step 3 — client-side PAD fast-guard. `moire_rgb` is the max
    // high-freq energy outside the skin-texture band across R/G/B FFTs
    // on the face ROI, computed at ~5 Hz in WebGL2 (OffscreenCanvas
    // fallback). Range roughly [0, 1]; higher = more screen-recapture
    // suspicion. The server's moire_branch cross-validates against this.
    padFast: { moire_rgb: 0.0, moire_ts: 0 },
};
const serverState = {
    hr: new Map(), spo2: new Map(), deception: new Map(),
    llm: new Map(), radar: {}, voice: {}, gsr: null,
    lastSeen: new Map(),
};
window._v93 = {instant, serverState, FF};

// v9.4.1 step 5 — per-person PAD verdict history for hysteresis-driven
// masking of HR / deception / LLM pills. Keys by pid → list of recent
// verdicts (max 6). `padRecentSpoofRatio(pid)` returns the fraction of
// SPOOF verdicts in the last 6 samples; the mask activates at ≥ 0.5.
const _padVerdictHistory = new Map();
function padRecordVerdict(pid, verdict){
    if(pid === undefined || pid === null) return;
    let h = _padVerdictHistory.get(pid);
    if(!h){ h = []; _padVerdictHistory.set(pid, h); }
    h.push(verdict || 'UNKNOWN');
    if(h.length > 6) h.shift();
}
function padRecentSpoofRatio(pid){
    const h = _padVerdictHistory.get(pid);
    if(!h || h.length === 0) return 0;
    let n = 0;
    for(const v of h){ if(v === 'SPOOF') n++; }
    return n / h.length;
}
function padShouldMask(pid, latestVerdict){
    if(!FF || !FF.PAD_UI) return false;
    if(latestVerdict !== 'SPOOF') return false;
    return padRecentSpoofRatio(pid) >= 0.5;
}
window._v93.pad = {
    history: _padVerdictHistory,
    recordVerdict: padRecordVerdict,
    recentSpoofRatio: padRecentSpoofRatio,
    shouldMask: padShouldMask,
};

// Step 10 — Latency HUD scaffolding. Performance marks + p50/p95/p99.
// When FF.LAT_HUD is on, the rAF loop + onmessage emit marks that the HUD
// panel summarizes once per second. Marks exist regardless of UI visibility
// so the /status endpoint can read them via performance.getEntriesByName.
const _latMarks = { instant: [], server: [], corr: [], hr: [], spo2: [], decep: [], llm: [], vitals: [] };
function _latObserve(source, latencyMs){
    const bucket = _latMarks[source];
    if(!bucket) return;
    bucket.push(latencyMs);
    if(bucket.length > 600) bucket.shift();  // 10 s @ 60 Hz
}
function _latPct(source, p){
    const b = _latMarks[source]; if(!b || b.length === 0) return null;
    const s = b.slice().sort((a,z)=>a-z);
    return s[Math.min(s.length-1, Math.floor(s.length*p/100))];
}
window._v93.lat = {observe: _latObserve, pct: _latPct, marks: _latMarks};

// Step 10 — Back-pressure state. If the ws:apply p95 stays above 50 ms
// for 2 seconds, we shed the lowest-priority envelope classes.
const _backpressure = {
    lastCheck: 0,
    shedCorr: false,
    shedVitals: false,
};
function _backpressureTick(){
    if(!FF.BACKPRESSURE) return;
    const now = performance.now();
    if(now - _backpressure.lastCheck < 2000) return;
    _backpressure.lastCheck = now;
    const p95 = _latPct('server', 95);
    if(p95 !== null && p95 > 50){
        if(!_backpressure.shedCorr){ _backpressure.shedCorr = true; console.warn('[v9.3 BP] shedding corr envelopes (p95=' + p95.toFixed(1) + 'ms)'); }
        else if(!_backpressure.shedVitals){ _backpressure.shedVitals = true; console.warn('[v9.3 BP] shedding vitals envelopes'); }
    } else if(p95 !== null && p95 < 20){
        _backpressure.shedCorr = false;
        _backpressure.shedVitals = false;
    }
}

// Step 2 — MediaStreamTrackProcessor + WebCodecs capability probe.
const HAS_MSTP = typeof MediaStreamTrackProcessor !== 'undefined';
const HAS_WEBCODECS = typeof VideoFrame !== 'undefined';
const HAS_RVFC = HTMLVideoElement.prototype && 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
if(FF.MSTP_CAPTURE){
    console.log('[v9.3 step 2] capture probes:', {HAS_MSTP, HAS_WEBCODECS, HAS_RVFC});
}
// When the flag + probes all pass we install a rVFC-driven per-frame
// callback. On every frame the callback stamps instant.frameTs with the
// presentation time so the HUD can measure capture→present separately.
let _rvfcBound = false;
function _bindRVFC(){
    if(_rvfcBound || !HAS_RVFC || !FF.MSTP_CAPTURE) return;
    const videoEl = document.getElementById('video');
    if(!videoEl) return;
    _rvfcBound = true;
    const loop = (now, metadata) => {
        instant.frameTs = metadata && metadata.mediaTime ? metadata.mediaTime : now;
        if(FF.LAT_HUD && metadata && metadata.expectedDisplayTime){
            _latObserve('instant', metadata.expectedDisplayTime - now);
        }
        try { videoEl.requestVideoFrameCallback(loop); } catch(e){}
    };
    try { videoEl.requestVideoFrameCallback(loop); } catch(e){ _rvfcBound = false; }
}

// Step 3 — WebGPU overlay pipeline. Full single-drawArrays landmark
// rendering with a shader that places one quad per landmark from a
// packed Float32 VBO. Falls back to WebGL2 (_initWebGL from v9.2) if
// WebGPU is unavailable OR if adapter/device acquisition fails.
const HAS_WEBGPU = !!(navigator.gpu);
let _gpuAdapter = null, _gpuDevice = null, _gpuCtx = null;
let _gpuPipeline = null, _gpuVBO = null, _gpuFormat = null;
let _gpuReady = false;
const _GPU_MAX_LANDMARKS = 478 * 5;  // 5 faces × 478 points
async function _initWebGPUOverlay(){
    if(!FF.WEBGPU || !HAS_WEBGPU) return false;
    try {
        _gpuAdapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
        if(!_gpuAdapter) return false;
        _gpuDevice = await _gpuAdapter.requestDevice();
        const cv = document.getElementById('overlay');
        if(!cv) return false;
        _gpuCtx = cv.getContext('webgpu');
        if(!_gpuCtx) return false;
        _gpuFormat = navigator.gpu.getPreferredCanvasFormat();
        _gpuCtx.configure({
            device: _gpuDevice, format: _gpuFormat,
            alphaMode: 'premultiplied',
        });
        // Vertex: position (vec2, normalized [0,1]), colour (vec3).
        // Each landmark is one POINT primitive; the fragment shader
        // renders a small filled circle by discarding corner pixels.
        const shaderSrc = `
            struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
            @vertex
            fn vs_main(@location(0) pos: vec2<f32>, @location(1) col: vec3<f32>) -> VSOut {
                var out: VSOut;
                // Normalized canvas coords → NDC with Y flipped.
                let ndc = vec2<f32>(pos.x * 2.0 - 1.0, 1.0 - pos.y * 2.0);
                out.pos = vec4<f32>(ndc, 0.0, 1.0);
                out.col = col;
                return out;
            }
            @fragment
            fn fs_main(@location(0) col: vec3<f32>) -> @location(0) vec4<f32> {
                return vec4<f32>(col, 0.65);
            }
        `;
        const mod = _gpuDevice.createShaderModule({ code: shaderSrc });
        _gpuPipeline = _gpuDevice.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: mod, entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 20,  // vec2 pos + vec3 col = 5 × f32
                    stepMode: 'vertex',
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x2' },
                        { shaderLocation: 1, offset: 8,  format: 'float32x3' },
                    ],
                }],
            },
            fragment: {
                module: mod, entryPoint: 'fs_main',
                targets: [{
                    format: _gpuFormat,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    },
                }],
            },
            primitive: { topology: 'point-list' },
        });
        _gpuVBO = _gpuDevice.createBuffer({
            size: _GPU_MAX_LANDMARKS * 20,   // 5 faces × 478 × 20 bytes
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        _gpuReady = true;
        console.log('[v9.3 step 3] WebGPU overlay pipeline ready');
        return true;
    } catch(e) {
        console.warn('[v9.3 step 3] WebGPU init failed:', e);
        _gpuReady = false;
        return false;
    }
}
// Render into the overlay canvas via WebGPU. Returns true if it drew,
// false to signal the caller to use the WebGL2 / 2D fallback.
function _drawWebGPU(faces){
    if(!_gpuReady || !_gpuDevice || !_gpuCtx || !faces) return false;
    let total = 0;
    for(const f of faces){ if(f && f.lm) total += Math.min(f.lm.length, 478); }
    if(total === 0){
        // Still clear the canvas so previous frame's points don't persist.
        try {
            const enc = _gpuDevice.createCommandEncoder();
            const pass = enc.beginRenderPass({
                colorAttachments: [{
                    view: _gpuCtx.getCurrentTexture().createView(),
                    loadOp: 'clear', storeOp: 'store',
                    clearValue: {r:0, g:0, b:0, a:0},
                }],
            });
            pass.end();
            _gpuDevice.queue.submit([enc.finish()]);
        } catch(e) {}
        return true;
    }
    if(total > _GPU_MAX_LANDMARKS) total = _GPU_MAX_LANDMARKS;
    const buf = new Float32Array(total * 5);
    let k = 0;
    for(const f of faces){
        if(!f || !f.lm) continue;
        const df = f.df || {};
        const dm = (_latestWS && _latestWS.av && _latestWS.av.dominant_emotion)
                 || df.emotion?.dominant || 'neutral';
        const hex = (typeof ec !== 'undefined' && ec[dm]) || '#22c55e';
        const r = parseInt(hex.slice(1, 3), 16) / 255;
        const g = parseInt(hex.slice(3, 5), 16) / 255;
        const b = parseInt(hex.slice(5, 7), 16) / 255;
        const n = Math.min(f.lm.length, 478);
        for(let i = 0; i < n; i++){
            if(k >= total * 5) break;
            const lm = f.lm[i];
            const lx = Array.isArray(lm) ? lm[0] : lm.x;
            const ly = Array.isArray(lm) ? lm[1] : lm.y;
            buf[k++] = lx; buf[k++] = ly;
            buf[k++] = r;  buf[k++] = g;  buf[k++] = b;
        }
    }
    _gpuDevice.queue.writeBuffer(_gpuVBO, 0, buf.buffer, 0, k * 4);
    try {
        const enc = _gpuDevice.createCommandEncoder();
        const pass = enc.beginRenderPass({
            colorAttachments: [{
                view: _gpuCtx.getCurrentTexture().createView(),
                loadOp: 'clear', storeOp: 'store',
                clearValue: {r:0, g:0, b:0, a:0},
            }],
        });
        pass.setPipeline(_gpuPipeline);
        pass.setVertexBuffer(0, _gpuVBO);
        pass.draw(k / 5);
        pass.end();
        _gpuDevice.queue.submit([enc.finish()]);
        return true;
    } catch(e){
        console.warn('[v9.3 step 3] WebGPU draw failed, falling back:', e);
        _gpuReady = false;
        return false;
    }
}

// Step 4 — On-device MediaPipe FaceLandmarker. Lazy-loaded from CDN the
// first time FF.CLIENT_TRACKER is on. A successful init switches the
// capture loop to mode=2 (client-landmarks-only) which stops the server
// from running its own MediaPipe, per the v9.3 correction-channel design.
let _clientTracker = null;
let _clientTrackerRunning = false;
async function _initClientTracker(){
    if(!FF.CLIENT_TRACKER || _clientTracker) return _clientTracker;
    try {
        const mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs');
        const vision = await mod.FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm');
        _clientTracker = await mod.FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numFaces: 5,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
        });
        console.log('[v9.3 step 4] MediaPipe FaceLandmarker ready (GPU delegate)');
        // Start the per-frame detect loop once the tracker is ready.
        _startClientTrackerLoop();
    } catch(e){
        console.warn('[v9.3 step 4] client tracker load failed:', e);
        _clientTracker = null;
    }
    return _clientTracker;
}
// Per-frame detect loop — runs at the camera's rVFC cadence so detection
// v9.5.x — Client-side per-face analyzer. Ports analysis.py:
//   blendshape_emotion, detect_blink, get_iris_data, eye_gaze, head_pose
// so the emotion / blink / gaze / pupil panels update at camera-rVFC cadence
// (zero WS round-trip). The server still computes these — its values are
// merged back on top of our client estimates (smoother + fused emotion)
// whenever the WS envelope arrives.
const _CLIENT_BLINK_HIST = [];      // timestamps (ms) of recent blinks, for BPM
const _CLIENT_PERCLOS_RING = [];    // rolling eye-closed booleans
const _CLIENT_PREV_BLINK = {};      // per-face: was-closed last frame?
const _CLIENT_EYE_CLOSED_THRESH = 0.55;
const _CLIENT_PERCLOS_WINDOW = 90;  // ~3 s at 30 fps
const _CLIENT_BLINK_WINDOW_MS = 60000;

function _clientBsDict(blendshapesArr){
    const d = Object.create(null);
    if(!blendshapesArr) return d;
    for(const c of blendshapesArr){
        if(c && c.categoryName) d[c.categoryName] = c.score || 0;
    }
    return d;
}

function _clientBlendshapeEmotion(bs){
    const _g = (k)=>bs[k]||0;
    const smile = (_g('mouthSmileLeft') + _g('mouthSmileRight'))/2;
    const frown = (_g('mouthFrownLeft') + _g('mouthFrownRight'))/2;
    const brow_down = (_g('browDownLeft') + _g('browDownRight'))/2;
    const brow_up = _g('browInnerUp');
    const brow_outer = (_g('browOuterUpLeft') + _g('browOuterUpRight'))/2;
    const jaw = _g('jawOpen');
    const eye_wide = (_g('eyeWideLeft') + _g('eyeWideRight'))/2;
    const eye_squint = (_g('eyeSquintLeft') + _g('eyeSquintRight'))/2;
    const nose_sneer = (_g('noseSneerLeft') + _g('noseSneerRight'))/2;
    const mouth_press = (_g('mouthPressLeft') + _g('mouthPressRight'))/2;
    const cheek_sq = (_g('cheekSquintLeft') + _g('cheekSquintRight'))/2;
    const lip_stretch = (_g('mouthStretchLeft') + _g('mouthStretchRight'))/2;
    const mouth_pucker = _g('mouthPucker');
    const mouth_roll = (_g('mouthRollLower') + _g('mouthRollUpper'))/2;
    const blink_active = Math.max(_g('eyeBlinkLeft'), _g('eyeBlinkRight'));
    const eye_wide_adj = eye_wide * Math.max(0, 1.0 - blink_active * 2.5);

    const e = {};
    e.happy = Math.min(1, smile*1.6 + cheek_sq*0.6 + eye_squint*0.4);
    e.sad = Math.min(1, frown*1.3 + brow_up*0.4 + mouth_press*0.2 + mouth_pucker*0.2);
    e.angry = Math.min(1, brow_down*1.4 + nose_sneer*0.9 + mouth_press*0.5 + lip_stretch*0.2);
    const s_brow = Math.max(0, (brow_up+brow_outer)*0.5 - 0.08);
    const s_eyes = Math.max(0, eye_wide_adj - 0.25);
    const s_jaw = Math.max(0, jaw - 0.20);
    e.surprise = (s_brow>0.05 && s_eyes>0 && s_jaw>0)
        ? Math.min(1, s_brow*1.2 + s_eyes*1.0 + s_jaw*0.8)
        : Math.min(1, (s_brow+s_eyes+s_jaw)*0.2);
    const f_eyes = Math.max(0, eye_wide_adj - 0.20);
    const f_lip = Math.max(0, lip_stretch - 0.10);
    const f_raw = f_eyes*0.8 + Math.max(0, brow_up-0.1)*0.6 + f_lip*1.2 + Math.max(0, jaw-0.15)*0.2;
    e.fear = Math.min(1, f_lip>0.05 ? f_raw : f_raw*0.15);
    e.disgust = Math.min(1, nose_sneer*1.6 + frown*0.4 + brow_down*0.2 + mouth_roll*0.3);
    let peak = 0; for(const k in e) peak = Math.max(peak, e[k]);
    e.neutral = Math.max(0, 1.0 - peak*1.2);
    for(const k in e) e[k] = Math.round(e[k]*1000)/1000;
    return e;
}

function _clientDetectBlink(bs, faceIdx){
    const blinkVal = ((bs.eyeBlinkLeft||0) + (bs.eyeBlinkRight||0))/2;
    const wasClosed = !!_CLIENT_PREV_BLINK[faceIdx];
    const isClosed = blinkVal > _CLIENT_EYE_CLOSED_THRESH;
    const blinked = wasClosed && !isClosed;
    _CLIENT_PREV_BLINK[faceIdx] = isClosed;
    const nowMs = performance.now();
    if(blinked) _CLIENT_BLINK_HIST.push(nowMs);
    while(_CLIENT_BLINK_HIST.length && nowMs - _CLIENT_BLINK_HIST[0] > _CLIENT_BLINK_WINDOW_MS)
        _CLIENT_BLINK_HIST.shift();
    const bpm = _CLIENT_BLINK_HIST.length;
    _CLIENT_PERCLOS_RING.push(isClosed);
    if(_CLIENT_PERCLOS_RING.length > _CLIENT_PERCLOS_WINDOW) _CLIENT_PERCLOS_RING.shift();
    const perclos = _CLIENT_PERCLOS_RING.length
        ? _CLIENT_PERCLOS_RING.filter(x=>x).length / _CLIENT_PERCLOS_RING.length : 0;
    const drowsiness = perclos>0.30 ? 'drowsy' : perclos>0.15 ? 'fatigued' : 'alert';
    return {
        blinked, blink_rate_pm: bpm,
        eye_openness: Math.round((1 - blinkVal)*100)/100,
        perclos: Math.round(perclos*1000)/1000,
        drowsiness,
    };
}

function _clientHeadPose(fl){
    const chin=fl[152], leftEye=fl[33], rightEye=fl[263], forehead=fl[10];
    const eye_dx = rightEye.x - leftEye.x, eye_dz = (rightEye.z||0) - (leftEye.z||0);
    const yaw = Math.atan2(eye_dz, eye_dx + 1e-6) * 180/Math.PI;
    const vert_dy = chin.y - forehead.y, vert_dz = (chin.z||0) - (forehead.z||0);
    const pitch = Math.atan2(vert_dz, vert_dy + 1e-6) * 180/Math.PI;
    const roll = Math.atan2(rightEye.y-leftEye.y, rightEye.x-leftEye.x + 1e-6) * 180/Math.PI;
    return {pitch: Math.round(pitch*10)/10, yaw: Math.round(yaw*10)/10,
            roll: Math.round(roll*10)/10};
}

function _clientEyeGaze(fl){
    try {
        const l_inner=fl[133], l_outer=fl[33], l_iris=fl[468];
        const r_inner=fl[362], r_outer=fl[263], r_iris=fl[473];
        const l_range = l_inner.x - l_outer.x;
        const l_pos = (l_iris.x - l_outer.x) / (l_range + 1e-6);
        const r_range = r_inner.x - r_outer.x;
        const r_pos = (r_iris.x - r_outer.x) / (r_range + 1e-6);
        const h_gaze = Math.round(((l_pos + r_pos)/2 - 0.5) * 2 * 100)/100;
        const l_top=fl[159], l_bot=fl[145];
        const l_v_range = l_bot.y - l_top.y;
        const l_v_pos = (l_iris.y - l_top.y) / (l_v_range + 1e-6);
        const r_top=fl[386], r_bot=fl[374];
        const r_v_range = r_bot.y - r_top.y;
        const r_v_pos = (r_iris.y - r_top.y) / (r_v_range + 1e-6);
        const v_gaze = Math.round(((l_v_pos + r_v_pos)/2 - 0.5) * 2 * 100)/100;
        return {horizontal: h_gaze, vertical: v_gaze};
    } catch(e){ return {horizontal: 0, vertical: 0}; }
}

function _clientIrisData(fl){
    try {
        const pt = (i)=>({x: Math.round(fl[i].x*1e4)/1e4, y: Math.round(fl[i].y*1e4)/1e4});
        const l_iris = pt(468), r_iris = pt(473);
        const l_ring = [468,469,470,471,472].map(pt);
        const r_ring = [473,474,475,476,477].map(pt);
        const l_eye = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246].map(pt);
        const r_eye = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398].map(pt);
        const l_eye_w = Math.abs(fl[33].x - fl[133].x);
        const l_iris_r = Math.abs(fl[469].x - fl[471].x) / 2;
        const r_eye_w = Math.abs(fl[362].x - fl[263].x);
        const r_iris_r = Math.abs(fl[474].x - fl[476].x) / 2;
        const l_dil = Math.round((l_iris_r/(l_eye_w+1e-6))*1000)/1000;
        const r_dil = Math.round((r_iris_r/(r_eye_w+1e-6))*1000)/1000;
        return {
            l_iris, r_iris, l_ring, r_ring, l_eye, r_eye,
            l_dilation: l_dil, r_dilation: r_dil,
            avg_dilation: Math.round(((l_dil + r_dil)/2)*1000)/1000,
        };
    } catch(e){ return null; }
}

// happens in lock-step with presentation. Populates `instant.faces` so
// the capture path can forward lmk to the server (mode=2).
function _startClientTrackerLoop(){
    if(_clientTrackerRunning) return;
    const videoEl = document.getElementById('video');
    if(!videoEl) return;
    _clientTrackerRunning = true;
    const loop = (now, metadata) => {
        if(!_clientTracker || !on){
            try { videoEl.requestVideoFrameCallback(loop); } catch(e){}
            return;
        }
        try {
            const res = _clientTracker.detectForVideo(videoEl, now);
            const faces = [];
            const lmks = res && res.faceLandmarks ? res.faceLandmarks : [];
            const blends = res && res.faceBlendshapes ? res.faceBlendshapes : [];
            for(let i = 0; i < lmks.length; i++){
                const raw = lmks[i];
                const flat = new Float32Array(raw.length * 3);
                for(let j = 0; j < raw.length; j++){
                    flat[j*3]     = raw[j].x;
                    flat[j*3 + 1] = raw[j].y;
                    flat[j*3 + 2] = raw[j].z || 0;
                }
                // v9.5.x — compute per-face analysis locally so the panels
                // (emotion / blink / pupil / gaze / pose) update at rVFC
                // cadence without any WS round-trip.
                const bsDict = _clientBsDict(blends[i] ? blends[i].categories : []);
                const bsEmo = _clientBlendshapeEmotion(bsDict);
                // Pick dominant emotion from the bs scores
                let dom = 'neutral', domV = 0;
                for(const k in bsEmo){ if(bsEmo[k] > domV){ domV = bsEmo[k]; dom = k; } }
                faces.push({
                    i: i,
                    pid: i,
                    track_id: i,
                    lm: raw.map(p => [p.x, p.y]),   // 2D points for draw
                    lmk: flat,                        // flat for WS send
                    blendshapes: blends[i] ? blends[i].categories : [],
                    bs_emotion: bsEmo,
                    fused_emotion: bsEmo,             // until server's fused arrives
                    _dominant_emotion: dom,
                    blink: _clientDetectBlink(bsDict, i),
                    pose: _clientHeadPose(raw),
                    gaze: _clientEyeGaze(raw),
                    iris: _clientIrisData(raw),
                    _client_computed: true,
                });
            }
            instant.faces = faces;
            instant.frameTs = now;
        } catch(e){
            if((instant._trackerErrBurst = (instant._trackerErrBurst || 0) + 1) % 60 === 0){
                console.warn('[v9.3 step 4] detectForVideo error:', e);
            }
        }
        try { videoEl.requestVideoFrameCallback(loop); } catch(e){}
    };
    try { videoEl.requestVideoFrameCallback(loop); } catch(e){ _clientTrackerRunning = false; }
}

// Step 6 — On-device ONNX emotion head. Disabled by default because the
// FER model has to be shipped separately (~4 MB) to /static/. If the
// model file is missing the loader quietly no-ops.
let _emoSession = null;
async function _initClientEmotion(){
    if(!FF.CLIENT_EMO || _emoSession) return _emoSession;
    try {
        const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.mjs');
        _emoSession = await ort.InferenceSession.create('/static/fer-mbnv3-int8.onnx', {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: 'all',
        });
        console.log('[v9.3 step 6] ONNX emotion head ready');
    } catch(e){
        console.warn('[v9.3 step 6] ONNX emotion head unavailable (model not found or ort not loaded):', e.message || e);
        _emoSession = null;
    }
    return _emoSession;
}

// Step 7 — MessagePack + Worker + SAB ring. The full SAB path requires
// COOP/COEP isolation (crossOriginIsolated === true). When unavailable
// we fall back to postMessage-transferred ArrayBuffer. Both paths parse
// msgpack in the worker so the main thread stays clean.
const HAS_CROSS_ORIGIN_ISOLATION = (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated);
let _wsWorker = null;
let _wsWorkerReady = false;
async function _initWSWorker(){
    if(!FF.SAB_RING || _wsWorker) return _wsWorker;
    try {
        const workerSrc = `
            // Minimal msgpack decoder — handles the subset we emit from
            // the server: maps, strings, ints, floats, arrays, booleans,
            // nil, bin8/16/32. Based on the msgpack spec (github).
            function decode(buf){
                const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
                const textDec = new TextDecoder();
                let pos = 0;
                function read(){
                    const b = view.getUint8(pos++);
                    if(b <= 0x7f) return b;
                    if(b >= 0xe0) return b - 0x100;
                    if(b >= 0xa0 && b <= 0xbf){ const n = b & 0x1f; const s = textDec.decode(new Uint8Array(buf.buffer, buf.byteOffset+pos, n)); pos += n; return s; }
                    if(b >= 0x90 && b <= 0x9f){ const n = b & 0x0f; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
                    if(b >= 0x80 && b <= 0x8f){ const n = b & 0x0f; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
                    switch(b){
                        case 0xc0: return null;
                        case 0xc2: return false;
                        case 0xc3: return true;
                        case 0xca: { const v = view.getFloat32(pos); pos += 4; return v; }
                        case 0xcb: { const v = view.getFloat64(pos); pos += 8; return v; }
                        case 0xcc: return view.getUint8(pos++);
                        case 0xcd: { const v = view.getUint16(pos); pos += 2; return v; }
                        case 0xce: { const v = view.getUint32(pos); pos += 4; return v; }
                        case 0xcf: { const v = Number(view.getBigUint64(pos)); pos += 8; return v; }
                        case 0xd0: { const v = view.getInt8(pos++); return v; }
                        case 0xd1: { const v = view.getInt16(pos); pos += 2; return v; }
                        case 0xd2: { const v = view.getInt32(pos); pos += 4; return v; }
                        case 0xd3: { const v = Number(view.getBigInt64(pos)); pos += 8; return v; }
                        case 0xd9: { const n = view.getUint8(pos++); const s = textDec.decode(new Uint8Array(buf.buffer, buf.byteOffset+pos, n)); pos += n; return s; }
                        case 0xda: { const n = view.getUint16(pos); pos += 2; const s = textDec.decode(new Uint8Array(buf.buffer, buf.byteOffset+pos, n)); pos += n; return s; }
                        case 0xdb: { const n = view.getUint32(pos); pos += 4; const s = textDec.decode(new Uint8Array(buf.buffer, buf.byteOffset+pos, n)); pos += n; return s; }
                        case 0xdc: { const n = view.getUint16(pos); pos += 2; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
                        case 0xdd: { const n = view.getUint32(pos); pos += 4; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
                        case 0xde: { const n = view.getUint16(pos); pos += 2; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
                        case 0xdf: { const n = view.getUint32(pos); pos += 4; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
                        default: throw new Error('msgpack type 0x' + b.toString(16) + ' not supported');
                    }
                }
                return read();
            }
            self.onmessage = (ev) => {
                const {type, buf} = ev.data || {};
                if(type === 'decode'){
                    try {
                        const obj = decode(new Uint8Array(buf));
                        self.postMessage({type: 'decoded', obj});
                    } catch(e){
                        self.postMessage({type: 'error', err: String(e)});
                    }
                }
            };
        `;
        const blobUrl = URL.createObjectURL(new Blob([workerSrc], {type:'application/javascript'}));
        _wsWorker = new Worker(blobUrl);
        _wsWorkerReady = true;
        console.log('[v9.3 step 7] WS msgpack worker ready. crossOriginIsolated=', HAS_CROSS_ORIGIN_ISOLATION);
    } catch(e){
        console.warn('[v9.3 step 7] WS worker init failed:', e);
        _wsWorker = null; _wsWorkerReady = false;
    }
    return _wsWorker;
}

// Step 8 — SolidJS signals. Lazy-loaded when FF.SOLID_SIGNALS flips on.
//
// Three pills are migrated as a proof of concept:
//   - HR (hrL / hrC)
//   - Oximeter SpO2 (oxiSp)
//   - Deception score
//
// When Solid is loaded, each pill is replaced by a Solid reactive text
// node that subscribes to its corresponding `serverState` signal. When
// a v9.3 envelope arrives and mutates the store, ONLY those specific
// text nodes re-render — no component-tree walk, no VDOM diff.
// Fall back silently when esm.sh is unreachable or CSP blocks the load.
let _solid = null, _solidStore = null, _setSolidStore = null;
async function _initSolid(){
    if(!FF.SOLID_SIGNALS) return null;
    try {
        const solid = await import('https://esm.sh/solid-js@1/dist/solid.js');
        const solidWeb = await import('https://esm.sh/solid-js@1/web/dist/web.js');
        const solidStore = await import('https://esm.sh/solid-js@1/store/dist/store.js').catch(()=>null);
        if(!solidStore){
            // Fall back to createSignal-only path
            const [get, set] = solid.createSignal({hr:null, spo2:null, decep:null});
            _solidStore = get; _setSolidStore = set;
        } else {
            const [st, setSt] = solidStore.createStore({hr:null, spo2:null, decep:null});
            _solidStore = st; _setSolidStore = setSt;
        }
        _solid = {solid, solidWeb};
        _mountSolidPills();
        console.log('[v9.3 step 8] SolidJS loaded, reactive pills mounted');
        return _solid;
    } catch(e){
        console.warn('[v9.3 step 8] SolidJS unavailable, keeping non-reactive pills:', e);
        return null;
    }
}
// Mount reactive pill components — each subscribes to its own signal
// and re-renders only when that signal changes. Pills are injected as
// children of existing DOM nodes so the overall layout doesn't shift.
function _mountSolidPills(){
    if(!_solid) return;
    const {createEffect} = _solid.solid;
    const pills = [
        {id: 'hrL-reactive', src: () => _solidStore.hr && _solidStore.hr.bpm, fmt: v => v == null ? '—' : String(Math.round(v))},
        {id: 'oxiSp-reactive', src: () => _solidStore.spo2 && _solidStore.spo2.spo2, fmt: v => v == null ? '—' : v + '%'},
        {id: 'decep-reactive', src: () => _solidStore.decep && _solidStore.decep.score, fmt: v => v == null ? '—' : v.toFixed(3)},
    ];
    for(const p of pills){
        const el = document.getElementById(p.id);
        if(!el) continue;
        createEffect(() => {
            const v = p.src();
            el.textContent = p.fmt(v);
        });
    }
}
// Bridge from serverState maps to the Solid store.
// Called whenever a new envelope arrives so Solid re-runs only the
// effect bound to that field.
function _pushToSolidStore(tag, payload){
    if(!_solidStore || !_setSolidStore) return;
    if(typeof _setSolidStore === 'function'){
        if(tag === 'hr')     _setSolidStore('hr',    payload);
        else if(tag === 'spo2')  _setSolidStore('spo2',  payload);
        else if(tag === 'decep') _setSolidStore('decep', payload);
    } else {
        // createSignal fallback
        const curr = typeof _solidStore === 'function' ? _solidStore() : _solidStore;
        const next = {...curr};
        if(tag === 'hr') next.hr = payload;
        if(tag === 'spo2') next.spo2 = payload;
        if(tag === 'decep') next.decep = payload;
        if(typeof _setSolidStore === 'function') _setSolidStore(next);
    }
}

// v9.3 step 1 — initialise every v9.3 subsystem with a guard. Any failure
// is logged but does NOT abort the page; the v9.2 path remains available.
(async function _v93Bootstrap(){
    if(FF.LAT_HUD) _bindRVFC();
    await Promise.allSettled([
        _initWebGPUOverlay(),
        _initClientTracker(),
        _initClientEmotion(),
        _initWSWorker(),
        _initSolid(),
    ]);
    // Kick off the latency HUD tick so counters populate even if nothing
    // draws yet.
    if(FF.LAT_HUD) setInterval(_updateLatHUD, 1000);
    if(FF.BACKPRESSURE) setInterval(_backpressureTick, 500);
})();

// Latency HUD — collapsible panel appended once the page loads. Shows
// per-source p50/p95/p99 over the last 10 seconds.
function _updateLatHUD(){
    let hud = document.getElementById('lat-hud');
    if(!hud){
        hud = document.createElement('aside');
        hud.id = 'lat-hud';
        hud.className = 'lat-hud';
        hud.style.cssText = 'position:fixed;right:6px;bottom:6px;background:#0a0a0a;border:1px solid #27272a;border-radius:6px;padding:6px 10px;font-family:monospace;font-size:10px;color:#a1a1aa;z-index:10000;min-width:280px';
        hud.innerHTML = '<div style="font-weight:700;color:#06b6d4;margin-bottom:4px">v9.3 LATENCY HUD  <span id="lat-hud-close" style="float:right;cursor:pointer;color:#52525b">×</span></div><div id="lat-hud-body"></div>';
        const _closeBtn = hud.querySelector('#lat-hud-close');
        if(_closeBtn) _closeBtn.addEventListener('click', function(){ hud.style.display = 'none'; });
        document.body.appendChild(hud);
    }
    const body = document.getElementById('lat-hud-body');
    if(!body) return;
    const sources = ['instant', 'server', 'corr', 'hr', 'spo2', 'decep', 'llm', 'vitals'];
    const rows = sources.map(src => {
        const p50 = _latPct(src, 50); const p95 = _latPct(src, 95); const p99 = _latPct(src, 99);
        const fmt = v => v === null ? '—' : v.toFixed(1);
        const warn = (p95 !== null && p95 > 50) ? 'color:#ef4444' : '';
        return '<div style="display:flex;justify-content:space-between;'+warn+'"><span>'+src+'</span><span>'+fmt(p50)+' / '+fmt(p95)+' / '+fmt(p99)+' ms</span></div>';
    }).join('');
    const bp = (_backpressure.shedCorr?'[corr]':'') + (_backpressure.shedVitals?'[vitals]':'');
    body.innerHTML = rows + '<div style="margin-top:4px;color:#71717a;border-top:1px solid #27272a;padding-top:3px">p50 / p95 / p99 (10s) ' + (bp?'shedding: '+bp:'') + '</div>';
}

// v9.5.x — inline msgpack decoder used by ws.onmessage. Same grammar as
// the Worker-hosted decoder above; we duplicate it here because the
// WS_MULTIPLEX server default (feature_flags.py) sends msgpack and the
// main-thread onmessage handler needs to decode without a Worker hop.
function _msgpackDecodeInline(u8){
    const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const textDec = new TextDecoder();
    let pos = 0;
    function read(){
        const b = view.getUint8(pos++);
        if(b <= 0x7f) return b;
        if(b >= 0xe0) return b - 0x100;
        if(b >= 0xa0 && b <= 0xbf){ const n = b & 0x1f; const s = textDec.decode(new Uint8Array(u8.buffer, u8.byteOffset+pos, n)); pos += n; return s; }
        if(b >= 0x90 && b <= 0x9f){ const n = b & 0x0f; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
        if(b >= 0x80 && b <= 0x8f){ const n = b & 0x0f; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
        switch(b){
            case 0xc0: return null;
            case 0xc2: return false;
            case 0xc3: return true;
            case 0xca: { const v = view.getFloat32(pos); pos += 4; return v; }
            case 0xcb: { const v = view.getFloat64(pos); pos += 8; return v; }
            case 0xcc: return view.getUint8(pos++);
            case 0xcd: { const v = view.getUint16(pos); pos += 2; return v; }
            case 0xce: { const v = view.getUint32(pos); pos += 4; return v; }
            case 0xcf: { const v = Number(view.getBigUint64(pos)); pos += 8; return v; }
            case 0xd0: { const v = view.getInt8(pos++); return v; }
            case 0xd1: { const v = view.getInt16(pos); pos += 2; return v; }
            case 0xd2: { const v = view.getInt32(pos); pos += 4; return v; }
            case 0xd3: { const v = Number(view.getBigInt64(pos)); pos += 8; return v; }
            case 0xd9: { const n = view.getUint8(pos++); const s = textDec.decode(new Uint8Array(u8.buffer, u8.byteOffset+pos, n)); pos += n; return s; }
            case 0xda: { const n = view.getUint16(pos); pos += 2; const s = textDec.decode(new Uint8Array(u8.buffer, u8.byteOffset+pos, n)); pos += n; return s; }
            case 0xdb: { const n = view.getUint32(pos); pos += 4; const s = textDec.decode(new Uint8Array(u8.buffer, u8.byteOffset+pos, n)); pos += n; return s; }
            case 0xdc: { const n = view.getUint16(pos); pos += 2; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
            case 0xdd: { const n = view.getUint32(pos); pos += 4; const a = new Array(n); for(let i=0;i<n;i++) a[i] = read(); return a; }
            case 0xde: { const n = view.getUint16(pos); pos += 2; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
            case 0xdf: { const n = view.getUint32(pos); pos += 4; const o = {}; for(let i=0;i<n;i++){ const k = read(); o[k] = read(); } return o; }
            case 0xc4: { const n = view.getUint8(pos++); const s = new Uint8Array(u8.buffer, u8.byteOffset+pos, n); pos += n; return s; }
            case 0xc5: { const n = view.getUint16(pos); pos += 2; const s = new Uint8Array(u8.buffer, u8.byteOffset+pos, n); pos += n; return s; }
            case 0xc6: { const n = view.getUint32(pos); pos += 4; const s = new Uint8Array(u8.buffer, u8.byteOffset+pos, n); pos += n; return s; }
            default: throw new Error('msgpack type 0x' + b.toString(16) + ' not supported');
        }
    }
    return read();
}

// v9.7.1 — heartbeat / liveness state. The browser-side detector
// declares the WS dead if no message has arrived in
// `_WS_LIVENESS_TIMEOUT_MS`. Forces close → onclose → reconnect via
// exponential backoff (already in place).
let _lastWsRxAt = 0;
const _WS_LIVENESS_TIMEOUT_MS = 15000;
let _wsLivenessTimer = null;
function _wsLivenessCheck(){
    if(!wsReady) return;
    const since = performance.now() - _lastWsRxAt;
    if(_lastWsRxAt > 0 && since > _WS_LIVENESS_TIMEOUT_MS){
        console.warn('[WS] no message for '+Math.round(since)+'ms, forcing reconnect');
        try { ws && ws.close(4001, 'liveness timeout'); } catch(_){}
    }
}

// v9.7.1 — periodic /health/full poll → server / DF / drop pills.
// Polls every 5 s (cheap; the endpoint is dependency-free aggregation).
async function _pollHealth(){
    try {
        const r = await fetch('/health/full');
        if(!r.ok) return;
        const j = await r.json();
        const ident = j.identity || {};
        const dfBlock = (j.subsystems && j.subsystems.deepface) || {};
        const breaker = dfBlock.circuit_breaker || {};
        if(typeof sEl === 'function'){
            sEl('sServer',
                'SRV: ' + (j.version||'?') + ' #' + (ident.content_hash||'????') + ' up=' +
                Math.round((j.uptime_s||0)/60) + 'm pid=' + (ident.pid||'?'),
                ident.content_hash ? '#22c55e' : '#52525b');
            const dfAge = dfBlock.last_update_age_s;
            const dfTxt = breaker.open
                ? 'DF: BREAKER OPEN ('+Math.round(breaker.open_seconds_remaining||0)+'s)'
                : (dfBlock.connected
                    ? 'DF: ' + (dfBlock.entries||0) + 'e age=' + (dfAge!=null?Math.round(dfAge):'?') + 's'
                    : 'DF: idle');
            const dfCol = breaker.open ? '#ef4444'
                       : (dfBlock.connected ? '#22c55e' : '#52525b');
            sEl('sDF', dfTxt, dfCol);
        }
        // mark client-known server hash so we can warn the user if it
        // changes mid-session (server restarted while this tab was open).
        if(_clientServerHash && ident.content_hash &&
           _clientServerHash !== ident.content_hash){
            console.warn('[server] content hash changed: '
                + _clientServerHash + ' -> ' + ident.content_hash
                + ' — consider Ctrl+Shift+R');
        }
        if(ident.content_hash) _clientServerHash = ident.content_hash;
    } catch(_){}
}
let _clientServerHash = null;
let _healthPollTimer = null;
function _startHealthPoll(){
    if(_healthPollTimer !== null) return;
    _pollHealth();
    _healthPollTimer = setInterval(_pollHealth, 5000);
}
// Boot the poll loop after DOM is ready.
if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', _startHealthPoll);
} else {
    _startHealthPoll();
}

// v9.7.1 — event-log drawer wiring. Polls /events/tail every 3 s when
// open; auto-opens on warn+ events so the user sees backend warnings
// without having to crack DevTools. Click EVT in the top bar to toggle.
let _evDrawerOpen = false;
let _evPollTimer = null;
let _evLastSeenTs = 0;
let _evAutoPopThrottleAt = 0;
const _LEVEL_COLORS = {
    debug:'#52525b', info:'#a1a1aa', warn:'#f59e0b',
    error:'#ef4444', critical:'#fca5a5'
};
const _LEVEL_RANK = {debug:0, info:1, warn:2, error:3, critical:4};
function toggleEventDrawer(){
    _evDrawerOpen = !_evDrawerOpen;
    const d = document.getElementById('eventDrawer');
    if(d) d.style.display = _evDrawerOpen ? 'block' : 'none';
    if(_evDrawerOpen){
        _evRefresh();
        if(_evPollTimer === null){ _evPollTimer = setInterval(_evRefresh, 3000); }
    } else if(_evPollTimer !== null){
        clearInterval(_evPollTimer); _evPollTimer = null;
    }
}
async function _evRefresh(){
    try {
        const sel = document.getElementById('evLevelFilter');
        const lvl = sel ? sel.value : '';
        const url = '/events/tail?n=80' + (lvl ? '&level=' + encodeURIComponent(lvl) : '');
        const r = await fetch(url);
        if(!r.ok) return;
        const j = await r.json();
        const list = document.getElementById('eventList');
        if(!list) return;
        const rows = j.lines || [];
        const seenAttn = rows.some(row => (row.ts||0) > _evLastSeenTs &&
            (_LEVEL_RANK[row.level||'info']||0) >= _LEVEL_RANK.warn);
        if(rows.length){
            _evLastSeenTs = Math.max(_evLastSeenTs, ...rows.map(r => r.ts||0));
        }
        const html = rows.map(row => {
            const lvl = row.level || 'info';
            const col = _LEVEL_COLORS[lvl] || '#a1a1aa';
            const ts = (row.ts_iso || '').slice(11, 23);
            const tag = row.tag || '?';
            const msg = row.msg || '';
            const ctx = Object.entries(row).filter(
                ([k]) => !['ts','ts_iso','level','tag','msg'].includes(k)
            ).map(([k,v]) => k+'='+(typeof v==='string'?v:JSON.stringify(v))).join(' ');
            return '<div style="border-bottom:1px solid #18181b;padding:2px 0">' +
                '<span style="color:'+col+';font-weight:700">['+lvl.toUpperCase()+']</span>'+
                ' <span style="color:#52525b">'+ts+'</span>'+
                ' <span style="color:#71717a">'+tag+':</span>'+
                ' <span>'+msg.replace(/[<>]/g, '')+'</span>'+
                (ctx ? ' <span style="color:#52525b">{'+ctx.replace(/[<>]/g, '')+'}</span>' : '') +
                '</div>';
        }).join('');
        list.innerHTML = html || '<div style="color:#52525b">no events yet</div>';
        // Auto-pop on warn+ (rate-limited so we don't pop more than once
        // every 30 s — repeated errors would otherwise become annoying).
        if(seenAttn && !_evDrawerOpen){
            const now = performance.now();
            if(now - _evAutoPopThrottleAt > 30000){
                _evAutoPopThrottleAt = now;
                toggleEventDrawer();
            }
        }
    } catch(_){}
}
// Background poll once per 10 s even when closed so we can detect
// warn-level events and auto-pop.
setInterval(_evRefresh, 10000);

// v9.7.1 — per-tile warmup. Track when each face was first seen.
// While value is missing AND elapsed < _TILE_WARMUP_MS, render
// "warming Xs"; after the timeout, render "FAILED".
const _TILE_WARMUP_MS = 8000;
const _faceWarmupT = {};
function _warmupRender(faceIdx, val){
    if(val !== '?' && val !== '--' && val != null) return val;
    const t0 = _faceWarmupT[faceIdx];
    if(!t0){
        _faceWarmupT[faceIdx] = performance.now();
        return 'warming 0s';
    }
    const elapsed = performance.now() - t0;
    if(elapsed < _TILE_WARMUP_MS){
        return 'warming ' + Math.floor(elapsed/1000) + 's';
    }
    return 'FAILED';
}
function _warmupReset(faceIdx){
    if(_faceWarmupT[faceIdx]) delete _faceWarmupT[faceIdx];
}

function startWS(){
    const proto=location.protocol==='https:'?'wss:':'ws:';
    var _authQs = '';
    if (window.__DODO_AUTH_TOKEN__ != null) {
        _authQs = '?token=' + encodeURIComponent(window.__DODO_AUTH_TOKEN__);
    }
    ws=new WebSocket(proto+'//'+location.host+'/ws/detect'+_authQs);
    ws.binaryType='arraybuffer';
    _lastWsRxAt = performance.now();
    if(_wsLivenessTimer === null){
        _wsLivenessTimer = setInterval(_wsLivenessCheck, 5000);
    }
    ws.onmessage=async(ev)=>{
        _lastWsRxAt = performance.now();
        try{let d;
        if(ev.data instanceof ArrayBuffer||ev.data instanceof Blob){
            const buf=ev.data instanceof Blob?await ev.data.arrayBuffer():ev.data;
            // v9.5.x: server sends msgpack when WS_MULTIPLEX=1 (the default).
            // Use the inline msgpack decoder instead of naively decoding as
            // UTF-8 text (which produced � garbage and broke face tracking).
            d = _msgpackDecodeInline(new Uint8Array(buf));
        }else{d=JSON.parse(ev.data);}
        // v9.3 step 5 — dispatch multiplexed envelopes by `t` tag. Legacy
        // full frame payloads (t="frame" or absent) drop through to the
        // existing draw path below.
        const tag = d && d.t;
        // v9.5.x BUGFIX — only adopt `_latestWS` when this envelope IS a
        // full frame. Previously ANY envelope (hr, decep, spo2, vitals…)
        // overwrote _latestWS, wiping .faces for rAF draw → face mesh
        // flickered on/off every 100 ms as multiplex envelopes arrived.
        if(!tag || tag === 'frame'){
            _latestWS = d;
            _latestWSArrivedAt = performance.now();
            const ms=d.ms||0;fpsArr.push(ms);if(fpsArr.length>30)fpsArr.shift();
        }
        const avgMs=fpsArr.length?Math.round(fpsArr.reduce((a,b)=>a+b,0)/fpsArr.length):0;
        const fps=avgMs?Math.round(1000/avgMs):0;
        if(tag && tag !== 'frame'){
            // Observe per-source latency
            if(d.ts_server_mono && FF.LAT_HUD){
                const age_ms = (performance.now() / 1000 - d.ts_server_mono) * 1000;
                _latObserve(tag, Math.max(0, age_ms));
            }
            if(tag === 'drop'){
                // v9.7.1 — server-side WS backpressure dropped this frame.
                // Update the DROP pill so the user sees overload pressure.
                _wsDropCount = (_wsDropCount||0) + 1;
                if(typeof sEl === 'function') sEl('sDrops', 'DROP: '+_wsDropCount,
                    _wsDropCount > 0 ? '#f59e0b' : '#52525b');
                return;  // no further processing for drop notices
            }
            if(tag === 'ping'){
                // v9.7.1 — server heartbeat. Already updated _lastWsRxAt
                // at top of onmessage; nothing else to do.
                return;
            }
            if(tag === 'hr')      { serverState.hr.set(d.pid ?? 0, d); _pushToSolidStore('hr', d); }
            else if(tag === 'spo2')  { serverState.spo2.set(d.pid ?? 0, d); _pushToSolidStore('spo2', d); }
            else if(tag === 'decep') { serverState.deception.set(d.pid ?? 0, d); _pushToSolidStore('decep', d); }
            else if(tag === 'llm')   serverState.llm.set(d.pid ?? 0, d);
            else if(tag === 'vitals') serverState.radar = d.radar || {};
            else if(tag === 'corr'){
                // v9.3 step 5 — correction policy: accept only if recent.
                const ageMs = d.ts_server_mono ? (performance.now()/1000 - d.ts_server_mono) * 1000 : 0;
                if(ageMs < 80 && (d.conf ?? 1) > 0.7 && !_backpressure.shedCorr){
                    // apply correction to instant.faces (stub — real tracker
                    // integration arrives in step 4 full wire-up)
                }
            }
            // v9.5.x FIX: do NOT decrement wsQueue on ancillary multiplex
            // envelopes. Only the primary t="frame" envelope counts as the
            // reply to the last _captureAndSend. The old code decremented
            // once per incoming envelope (5-6×/frame when multiplex is on),
            // driving wsQueue negative, defeating backpressure, and letting
            // the server's OS receive buffer build up 3-4 seconds of lag.
            return;
        }
        if(FF.LAT_HUD && d.ts_server_mono){
            const ageMs = (performance.now()/1000 - d.ts_server_mono) * 1000;
            _latObserve('server', Math.max(0, ageMs));
        }
        _pushLandmarkHistory(d);  // v9.2 step 19 — predictive interpolation
        if(d.av && d.av.oxi_spo2) drawSpo2Spark(d.av.oxi_spo2);
        if(d.ai&&d.ai.length){uAI(d.ai);d.ai.forEach(a=>{if(a.text)pushAIHistory(a.text);});}
        if(d.persons&&d.persons.length>0){uDeception(d.persons[0]);drawDecSpark(d.persons[0]?.deception?.score);}
        if(d.frame_quality)uQuality(d.frame_quality);
        if(d.calibration_lap)renderLapHud(d.calibration_lap);
        uSpeakers(d);
        // v9.7.5: PAIRS badge fix — was reading `d.faces.length` (number
        // of faces in frame), which is why it permanently showed 1 for
        // a single-face setup regardless of how many QLoRA training
        // pairs had actually been logged. The real count now comes
        // from /qlora/status polled at 1 Hz (see _qloraBadgeRefresh).
        // We leave this assignment as a no-op to avoid churning the
        // badge between WS frames; the poll-based refresh below owns
        // the value.
        // v9.7.6 — clarify what "fps" actually means in the top-right pill.
        //
        // The number we compute as `fps = 1000/avgMs` is the WebSocket
        // ENVELOPE arrival rate (how often the server sends us updates).
        // It's not the camera frame rate. Users seeing "20 fps" here while
        // the boot log said the high-fps cam runs at 59 fps assumed the
        // camera was broken — it wasn't, the analysis pipeline was just
        // throttled to ~20 Hz publication. We now label this `ws` and
        // add the actual dual-camera FPS (`cam`) right next to it so
        // both rates are visible. "ws Hz" is also more honest than "fps"
        // since envelopes can carry multiple frames' worth of analysis.
        const _camFps = (d.dual_cam && d.dual_cam.fps) ? Math.round(d.dual_cam.fps) : null;
        const _camStr = (_camFps != null) ? (' | cam:' + _camFps + 'fps') : '';
        document.getElementById('stats').textContent='ws:'+fps+'Hz | '+avgMs+'ms'+_camStr+' | '+d.mode+' | Q:'+Math.round(_jpegQ*100)+'%';
        _adaptJpegQ(avgMs);
        wsQueue--;}catch(e){console.error('[WS] Parse error:',e);wsQueue--;}
    };
    ws.onerror=(e)=>{wsReady=false;console.log('[WS] Error, falling back to HTTP');document.getElementById('stats').textContent='WebSocket failed, using HTTP fallback...';document.getElementById('stats').style.color='#f59e0b';ws=null;lpHTTP();};
    ws.onopen=function(){wsReady=true;wsQueue=0;_wsBackoff=1000;console.log('[WS] Connected');document.getElementById('stats').textContent='WebSocket connected, sending frames...';document.getElementById('stats').style.color='#22c55e';lp();};
    ws.onclose=()=>{wsReady=false;if(on){const jitter=Math.random()*1000-500;const delay=Math.min(30000,_wsBackoff)+jitter;console.log('[WS] Closed, reconnecting in '+Math.round(delay)+'ms');setTimeout(startWS,delay);_wsBackoff=Math.min(30000,_wsBackoff*2);}};
}
// v8.6: dual-cadence loop.
//   - FULL (heavy DeepFace + per-person LLM-relevant) frame fires every 1000ms (1Hz)
//     so emotional/vital analysis is GUARANTEED to run every second.
//   - FAST (MediaPipe-only, cheap) frames fire in between for smooth overlay/HR.
// Both share the WS connection but the full-cadence is wall-clock anchored.
let _lastFullSent=0;
let _lastFastSent=0;
// v9.1.5: native-refresh-rate capture loop using requestAnimationFrame.
// The old setTimeout-based scheduler introduced 8-33 ms jitter on top of
// whatever the WS round-trip was. rAF locks the capture to the display's
// VSYNC tick (60/120/144 Hz) so every UI paint has the freshest possible
// frame — the face mask overlay now tracks the face to the pixel with
// zero visible "trail" even during rapid head turns.
//
// FULL_INTERVAL_MS stays at 500 ms (DeepFace emotion / demographics are
// the expensive path — cheaper to run at 2 Hz and let fast frames fill
// in between). MAX_INFLIGHT lowered to 2 so the queue can never hold
// more than two unanswered frames, which kept latency bounded to
// 2 × server-processing-time.
const FULL_INTERVAL_MS=500;
const FAST_MIN_INTERVAL_MS=4;    // hard floor — allows up to ~120 fps on high-refresh displays
// v9.2 (plan step 17 & 20): persistent capture canvas + encoding tiers.
// Tier 1 — Worker + OffscreenCanvas (fully off main thread).
// Tier 2 — OffscreenCanvas.convertToBlob() (off-thread encoding only).
// Tier 3 — canvas.toBlob (main thread, legacy fallback).
let _capCanvas = null;
let _capOffscreen = null;
let _capWorker = null;
let _capTier = 'toBlob';         // diagnostic label
let _workerPending = null;       // Promise resolver when worker is busy
function _getCaptureCanvas(){
    if(_capCanvas) return;
    _capCanvas = document.createElement('canvas');
    _capCanvas.width = W; _capCanvas.height = H;
    // Tier 1 — try Worker first
    if(typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined'){
        try {
            const workerSrc = `
                let oc = null;
                self.onmessage = async (ev) => {
                    const {type, bitmap, quality, frameId} = ev.data;
                    if(type === 'init'){
                        oc = new OffscreenCanvas(${W}, ${H});
                        self.postMessage({type:'ready'});
                        return;
                    }
                    if(type === 'encode' && bitmap){
                        try {
                            const ctx = oc.getContext('2d');
                            ctx.drawImage(bitmap, 0, 0, ${W}, ${H});
                            bitmap.close && bitmap.close();
                            const blob = await oc.convertToBlob({type:'image/jpeg', quality});
                            const buf = await blob.arrayBuffer();
                            self.postMessage({type:'encoded', buf, frameId}, [buf]);
                        } catch(e){
                            self.postMessage({type:'error', frameId, err: String(e)});
                        }
                    }
                };
            `;
            const blobUrl = URL.createObjectURL(new Blob([workerSrc], {type:'application/javascript'}));
            _capWorker = new Worker(blobUrl);
            _capWorker.postMessage({type:'init'});
            _capWorker.onmessage = (ev) => {
                if(ev.data.type === 'ready'){ _capTier = 'worker'; console.log('[capture] Tier 1: Worker + OffscreenCanvas (fully off-main-thread)'); }
                else if(ev.data.type === 'encoded'){
                    if(_workerPending){ _workerPending(ev.data.buf); _workerPending = null; }
                }
                else if(ev.data.type === 'error'){
                    if(_workerPending){ _workerPending(null); _workerPending = null; }
                }
            };
        } catch(e) {
            console.warn('[capture] worker init failed, falling back:', e);
            _capWorker = null;
        }
    }
    // Tier 2 — OffscreenCanvas.convertToBlob on main thread
    if(!_capWorker && typeof OffscreenCanvas !== 'undefined'){
        try {
            _capOffscreen = new OffscreenCanvas(W, H);
            if(typeof _capOffscreen.convertToBlob === 'function'){
                _capTier = 'offscreen';
                console.log('[capture] Tier 2: OffscreenCanvas.convertToBlob (off-main-thread JPEG)');
            } else { _capOffscreen = null; }
        } catch(e) { _capOffscreen = null; }
    }
    if(_capTier === 'toBlob') console.log('[capture] Tier 3: canvas.toBlob (legacy main-thread fallback)');
}
async function _encodeBlob(ctxSrc, quality){
    // Tier 1 — Worker + OffscreenCanvas: grab an ImageBitmap, ship to worker.
    if(_capWorker){
        try {
            const bitmap = await createImageBitmap(ctxSrc);
            return await new Promise(resolve => {
                _workerPending = (buf) => resolve(buf ? new Blob([buf], {type:'image/jpeg'}) : null);
                _capWorker.postMessage({type:'encode', bitmap, quality, frameId: performance.now()}, [bitmap]);
            });
        } catch(e) {
            // fall through to tier 2/3
        }
    }
    // Tier 2 — OffscreenCanvas off-thread encode.
    if(_capOffscreen){
        const octx = _capOffscreen.getContext('2d');
        octx.drawImage(ctxSrc, 0, 0, W, H);
        return await _capOffscreen.convertToBlob({type:'image/jpeg', quality: quality});
    }
    // Tier 3 — legacy toBlob.
    return await new Promise(r => _capCanvas.toBlob(r, 'image/jpeg', quality));
}
// v9.3.1 step 4 / v9.4.1 step 3 — build the v9.3 binary header.
// Format (matches _parse_v93_header in main.py):
//   byte  0   mode        0=fast, 1=full, 2=client_landmarks_only
//   byte  1   flags       bit0=has_jpeg, bit1=has_lmk, bit2=wants_correction,
//                          bit3=magic, bit4=has_padfast (v9.4.1)
//   u16LE 2   lmk_count
//   u32BE 4   jpeg_len
//   u32LE 8   pid
//   payload   [lmk Float32 lmk_count*3] then [jpeg] then [padfast f32 if flag]
function _buildV93Header(mode, lmkArray, jpegBuf, pid, padFastMoire){
    const hasJpeg = jpegBuf && jpegBuf.byteLength > 0;
    const hasLmk = lmkArray && lmkArray.length > 0;
    const hasPadFast = (typeof padFastMoire === 'number' && padFastMoire >= 0);
    let flags = 0x08;                              // magic
    if(hasJpeg) flags |= 0x01;
    if(hasLmk)  flags |= 0x02;
    if(hasLmk)  flags |= 0x04;                     // wants_correction
    if(hasPadFast) flags |= 0x10;                   // v9.4.1: trailing f32
    const lmkCount = hasLmk ? (lmkArray.length / 3) : 0;
    const jpegLen = hasJpeg ? jpegBuf.byteLength : 0;
    const hdrLen = 12;
    const lmkBytes = hasLmk ? (lmkCount * 3 * 4) : 0;
    const padFastBytes = hasPadFast ? 4 : 0;
    const total = hdrLen + lmkBytes + jpegLen + padFastBytes;
    const out = new ArrayBuffer(total);
    const dv = new DataView(out);
    const u8 = new Uint8Array(out);
    dv.setUint8(0, mode);
    dv.setUint8(1, flags);
    dv.setUint16(2, lmkCount, true);               // little-endian
    dv.setUint32(4, jpegLen, false);               // big-endian, legacy
    dv.setUint32(8, pid >>> 0, true);
    if(hasLmk){
        const fView = new Float32Array(out, hdrLen, lmkCount * 3);
        fView.set(lmkArray);
    }
    if(hasJpeg){
        u8.set(new Uint8Array(jpegBuf), hdrLen + lmkBytes);
    }
    if(hasPadFast){
        dv.setFloat32(hdrLen + lmkBytes + jpegLen, padFastMoire, true);
    }
    return out;
}

// ─── v9.4.1 step 3 — WebGL2 client-side moiré fast-guard ──────────────
// Samples a 64×64 face-ROI patch from the <video> element at ~5 Hz,
// computes |FFT| per R/G/B channel via an OffscreenCanvas 2D path, and
// writes the max high-freq energy outside the skin-texture band to
// instant.padFast.moire_rgb. The server's moire_branch reads this via
// the v9.3 binary envelope (flag 0x10, trailing float32).
//
// The shader path is a cooperative 2D DFT (no iFFT needed); for
// browsers without WebGL2 we fall back to OffscreenCanvas pixel reads
// and a small JS radix-2 FFT. Runtime budget is ≤ 2 ms/tick on an
// integrated GPU.
const _PAD_FAST_TICK_MS = 200;
let _padFastLastTick = 0;
let _padFastCanvas = null;
let _padFastCtx = null;
const _PAD_FAST_SIZE = 64;

function _padFastInit(){
    if(_padFastCanvas) return;
    try {
        _padFastCanvas = new OffscreenCanvas(_PAD_FAST_SIZE, _PAD_FAST_SIZE);
        _padFastCtx = _padFastCanvas.getContext('2d', {willReadFrequently: true});
    } catch(e){
        _padFastCanvas = document.createElement('canvas');
        _padFastCanvas.width = _PAD_FAST_SIZE;
        _padFastCanvas.height = _PAD_FAST_SIZE;
        _padFastCtx = _padFastCanvas.getContext('2d', {willReadFrequently: true});
    }
}

function _fft1d(re, im){
    // In-place radix-2 Cooley-Tukey. Sizes must be power of 2.
    const n = re.length;
    let j = 0;
    for(let i=1; i<n; i++){
        let bit = n >> 1;
        for(; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if(i < j){
            let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }
    for(let len = 2; len <= n; len <<= 1){
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang), wIm = Math.sin(ang);
        for(let i = 0; i < n; i += len){
            let pRe = 1, pIm = 0;
            for(let k = 0; k < (len >> 1); k++){
                const a = i + k;
                const b = a + (len >> 1);
                const tRe = pRe * re[b] - pIm * im[b];
                const tIm = pRe * im[b] + pIm * re[b];
                re[b] = re[a] - tRe;
                im[b] = im[a] - tIm;
                re[a] += tRe;
                im[a] += tIm;
                const npRe = pRe * wRe - pIm * wIm;
                pIm = pRe * wIm + pIm * wRe;
                pRe = npRe;
            }
        }
    }
}

function _highFreqRatio(channel){
    // channel is a Float32Array of length 64*64. Returns [0,1] ratio of
    // high-frequency energy outside a central skin-texture band over
    // total magnitude. DC removed before FFT.
    const N = _PAD_FAST_SIZE;
    let mean = 0;
    for(let i=0; i<channel.length; i++) mean += channel[i];
    mean /= channel.length;
    const re = new Float32Array(N * N);
    const im = new Float32Array(N * N);
    for(let i=0; i<channel.length; i++) re[i] = channel[i] - mean;
    // 2D FFT via row + column passes (shared temp buffers)
    const rowRe = new Float32Array(N);
    const rowIm = new Float32Array(N);
    for(let r=0; r<N; r++){
        for(let c=0; c<N; c++){ rowRe[c] = re[r*N+c]; rowIm[c] = im[r*N+c]; }
        _fft1d(rowRe, rowIm);
        for(let c=0; c<N; c++){ re[r*N+c] = rowRe[c]; im[r*N+c] = rowIm[c]; }
    }
    for(let c=0; c<N; c++){
        for(let r=0; r<N; r++){ rowRe[r] = re[r*N+c]; rowIm[r] = im[r*N+c]; }
        _fft1d(rowRe, rowIm);
        for(let r=0; r<N; r++){ re[r*N+c] = rowRe[r]; im[r*N+c] = rowIm[r]; }
    }
    // Magnitude shift: treat coord (r,c) with fftshift where (N/2,N/2) = DC
    let total = 1e-9, hi = 0;
    const half = N / 2;
    // Skin-texture band: normalized freq ≤ 0.2 (low-freq, smooth skin)
    // High band: normalized freq ≥ 0.4
    for(let r=0; r<N; r++){
        const dr = r < half ? r : r - N;
        for(let c=0; c<N; c++){
            const dc = c < half ? c : c - N;
            const fnorm = Math.sqrt(dr*dr + dc*dc) / half;  // 0..sqrt(2)
            const mag = Math.hypot(re[r*N+c], im[r*N+c]);
            total += mag;
            if(fnorm > 0.4) hi += mag;
        }
    }
    return Math.min(1.0, hi / total);
}

function _padFastComputeMoire(){
    // Called from the capture loop. Writes instant.padFast.moire_rgb.
    if(!FF.PAD_MOIRE || !FF.PAD_UI) return;
    const now = performance.now();
    if(now - _padFastLastTick < _PAD_FAST_TICK_MS) return;
    _padFastLastTick = now;
    const v = document.getElementById('video');
    if(!v || !v.videoWidth || !v.videoHeight) return;
    _padFastInit();
    if(!_padFastCtx) return;
    // Draw a centred face-sized ROI into the 64x64 canvas. Without a
    // face bbox we fall back to the centre of the video frame.
    let sx, sy, sw, sh;
    const f = instant.faces && instant.faces[0];
    if(f && f.bbox && f.bbox.length === 4){
        [sx, sy, sw, sh] = f.bbox;
    } else {
        const vw = v.videoWidth, vh = v.videoHeight;
        const side = Math.min(vw, vh) * 0.5;
        sx = (vw - side) / 2;
        sy = (vh - side) / 2;
        sw = sh = side;
    }
    try {
        _padFastCtx.drawImage(v, sx, sy, sw, sh, 0, 0,
                              _PAD_FAST_SIZE, _PAD_FAST_SIZE);
        const img = _padFastCtx.getImageData(0, 0, _PAD_FAST_SIZE, _PAD_FAST_SIZE);
        const px = img.data;
        const N = _PAD_FAST_SIZE * _PAD_FAST_SIZE;
        const rCh = new Float32Array(N);
        const gCh = new Float32Array(N);
        const bCh = new Float32Array(N);
        for(let i=0; i<N; i++){
            rCh[i] = px[4*i] / 255;
            gCh[i] = px[4*i + 1] / 255;
            bCh[i] = px[4*i + 2] / 255;
        }
        const rr = _highFreqRatio(rCh);
        const gr = _highFreqRatio(gCh);
        const br = _highFreqRatio(bCh);
        const peak = Math.max(rr, gr, br);
        instant.padFast.moire_rgb = peak;
        instant.padFast.moire_ts = now;
    } catch(e){
        // Intentionally quiet — the fast-guard is opt-in and best-effort.
    }
}

// v9.3.1 step 4 — capture-and-send, now aware of the client tracker.
// When FF.CLIENT_TRACKER is on AND _clientTracker is ready AND the last
// frame produced landmarks, send mode=2 with the lmk array; the server
// will skip its own MediaPipe pass. Otherwise fall back to legacy
// mode=0/1 (JPEG-only). The v9.3 binary header is always used when
// FF.WS_MULTIPLEX is on; legacy one-byte header otherwise.
function _captureAndSend(){
    if(!on||!wsReady) return;
    if(wsQueue >= MAX_INFLIGHT) return;
    const now = performance.now();
    const dueFast = (now - _lastFastSent) >= FAST_MIN_INTERVAL_MS;
    const dueFull = (now - _lastFullSent) >= FULL_INTERVAL_MS;
    if(!(dueFast||dueFull)) return;
    wsQueue++;
    _getCaptureCanvas();
    const _t0 = now;
    // v9.4.1 step 3: run the client-side moiré fast-guard (rate-limited).
    _padFastComputeMoire();
    const haveClientLmk = (FF.CLIENT_TRACKER && _clientTracker && instant.faces.length > 0);
    // v9.5.x BUGFIX — when client-tracker is on, we were ALWAYS sending
    // mode=2 → the server's _process_frame_full never fired → DeepFace
    // (age, gender, race), fused emotion, and the full deception feed
    // all sat empty. Fix: at the FULL_INTERVAL_MS cadence (500 ms) force
    // mode=1 so the server runs its heavy path too; all fast frames in
    // between stay on mode=2 for zero-RTT tracking.
    let mode;
    if(dueFull){
        mode = 1;   // Heavy full frame — server does DeepFace + emotion fusion.
    } else if(haveClientLmk){
        mode = 2;   // Fast frame with client lmk — server skips MediaPipe.
    } else {
        mode = 0;   // Fast frame, server does everything.
    }
    // v9.7.3 BUGFIX: previous version bumped _lastFullSent on mode===2 as
    // well, which meant once the client-side tracker was active (the
    // common case), `dueFull` was reset every frame and mode=1 never
    // fired again after the first frame. Result: DeepFace ran once at
    // boot then the cache expired → AGE/GENDER/ETHNICITY tiles
    // permanently said FAILED. Only mode=1 actually triggers the
    // server-side _process_frame_full path that submits DeepFace, so
    // only mode=1 should reset the dueFull timer.
    if(mode===1) _lastFullSent = now;
    _lastFastSent = now;
    // Draw the video frame into the reusable canvas — we still send JPEG
    // alongside lmk in mode=2 so rPPG ROI has pixels to sample.
    _capCanvas.getContext('2d').drawImage(v, 0, 0, W, H);
    const _q = (mode===1||mode===2) ? Math.min(0.7, _jpegQ+0.15) : _jpegQ;
    _encodeBlob(_capCanvas, _q).then(async blob => {
        if(!wsReady||!blob){wsQueue--;return;}
        const jpegBuf = await blob.arrayBuffer();
        let combined;
        if(FF.WS_MULTIPLEX){
            // Pack the client tracker landmarks (if any) into the v9.3 header.
            let lmk = null;
            if(haveClientLmk && instant.faces[0] && instant.faces[0].lmk){
                lmk = instant.faces[0].lmk;  // Float32Array of length N*3
            }
            const pid = (instant.faces[0] && instant.faces[0].pid) | 0;
            // v9.4.1 step 3: forward padFast.moire_rgb when stale ≤ 1 s.
            let padFast = -1;
            if(FF.PAD_MOIRE && FF.PAD_UI &&
               instant.padFast && instant.padFast.moire_ts &&
               (now - instant.padFast.moire_ts) < 1000){
                padFast = instant.padFast.moire_rgb;
            }
            const hdrBuf = _buildV93Header(mode, lmk, jpegBuf, pid,
                                          padFast >= 0 ? padFast : undefined);
            combined = new Uint8Array(hdrBuf);
        } else {
            const header = new Uint8Array([mode]);
            combined = new Uint8Array(1 + jpegBuf.byteLength);
            combined.set(header);
            combined.set(new Uint8Array(jpegBuf), 1);
        }
        try{ ws.send(combined); }catch(e){ wsQueue--; }
        n++; _totalFrames++;
        _adaptJpegQ(performance.now()-_t0);
        _updateSessionTimer();
    }).catch(()=>{ wsQueue--; });
}
// ═══════════════════════════════════════════════════════════════════
// v9.2 plan steps 17-30 — UI perf + number smoothness block
// ═══════════════════════════════════════════════════════════════════

// ─── Step 24: performance markers ────────────────────────────────────
// Lightweight wrapper around the User Timing API so bottlenecks are
// visible in Chrome DevTools Performance panel without any extra setup.
const _PERF_ENABLED = (typeof performance !== 'undefined' && !!performance.mark);
function _pmark(name){ if(_PERF_ENABLED) performance.mark(name); }
function _pmeasure(name, from, to){
    try { if(_PERF_ENABLED) performance.measure(name, from, to); } catch(e) {}
}

// ─── Step 19: predictive landmark interpolation ──────────────────────
// Keep the last N landmark frames so we can EXTRAPOLATE future positions
// based on velocity. This way the face mask tracks real-time head motion
// even when the server is only responding at 20-30 fps. We use the two
// newest frames for linear velocity and project forward by the elapsed
// time since the newest frame arrived.
const _LM_HISTORY_MAX = 3;
const _LM_EXTRAPOLATE_MS = 200;  // cap projection so a stale payload doesn't fly off
let _lmHistory = [];     // [{t_ms: number, faces: [...copy of d.faces for geometry]}]
function _pushLandmarkHistory(d){
    if(!d || !d.faces) return;
    // Keep only the geometry — lm/iris/pose/gaze — not the heavy data.
    const slim = d.faces.map(f => ({
        i: f.i, lm: f.lm, iris: f.iris, gaze: f.gaze, pose: f.pose, identity: f.identity,
        df: f.df, bs_emotion: f.bs_emotion, fused_emotion: f.fused_emotion,
    }));
    _lmHistory.push({t_ms: performance.now(), faces: slim});
    if(_lmHistory.length > _LM_HISTORY_MAX) _lmHistory.shift();
}
function _predictedFaces(nowMs){
    if(_lmHistory.length === 0) return null;
    const newest = _lmHistory[_lmHistory.length - 1];
    if(_lmHistory.length < 2) return newest.faces;
    const dt = nowMs - newest.t_ms;
    if(dt <= 0 || dt > _LM_EXTRAPOLATE_MS) return newest.faces;
    const prev = _lmHistory[_lmHistory.length - 2];
    const frameDt = newest.t_ms - prev.t_ms;
    if(frameDt <= 0) return newest.faces;
    const vel = dt / frameDt;   // extrapolation factor (0 → newest, 1 → +1 full frame)
    const out = [];
    for(let fi = 0; fi < newest.faces.length; fi++){
        const nf = newest.faces[fi];
        const pf = prev.faces[fi];
        if(!pf || !nf.lm || !pf.lm || nf.lm.length !== pf.lm.length){
            out.push(nf);
            continue;
        }
        // Linear velocity extrapolation on landmarks (lm is [[x,y], ...] or [{x,y}, ...]).
        const lm = new Array(nf.lm.length);
        for(let i = 0; i < nf.lm.length; i++){
            const a = nf.lm[i];
            const b = pf.lm[i];
            if(Array.isArray(a)){
                lm[i] = [a[0] + (a[0] - b[0]) * vel, a[1] + (a[1] - b[1]) * vel];
            } else {
                lm[i] = {x: a.x + (a.x - b.x) * vel, y: a.y + (a.y - b.y) * vel, z: a.z};
            }
        }
        out.push({...nf, lm});
    }
    return out;
}

// ─── Steps 25, 28, 29, 30: SmoothedNumber + dead-zone + lerp + unified class ─
// Every numeric display passes through this so updates are buttery and
// dead-zoned against noise.
class SmoothedNumber {
    // ms   = transition time between values (lerp duration)
    // dead = ignore updates smaller than this magnitude
    // precision = decimals to render
    constructor({ms=180, dead=0, precision=0, minDelta=0} = {}){
        this._from = null;       // starting value for current transition
        this._target = null;     // latest received value (destination)
        this._startT = 0;        // performance.now() when transition started
        this._ms = ms;
        this._dead = dead;
        this._minDelta = minDelta;
        this._precision = precision;
        this._lastRendered = null;
    }
    set(v){
        if(v == null || Number.isNaN(v)) return;
        const curr = this.peek();
        // Dead-zone: ignore sub-threshold updates.
        if(curr != null && this._dead > 0 && Math.abs(v - curr) < this._dead) return;
        // Min-delta from last rendered: same idea, stops flicker.
        if(this._lastRendered != null && this._minDelta > 0
           && Math.abs(v - this._lastRendered) < this._minDelta) return;
        this._from = curr == null ? v : curr;
        this._target = v;
        this._startT = performance.now();
    }
    peek(){
        if(this._target == null) return null;
        const t = performance.now() - this._startT;
        if(t >= this._ms) return this._target;
        const u = Math.max(0, Math.min(1, t / this._ms));
        // Ease-out cubic for a silky stop
        const e = 1 - Math.pow(1 - u, 3);
        return this._from + (this._target - this._from) * e;
    }
    render(){
        const v = this.peek();
        if(v == null) return null;
        const out = this._precision > 0 ? v.toFixed(this._precision) : Math.round(v);
        this._lastRendered = v;
        return out;
    }
    reset(){ this._from = this._target = null; this._lastRendered = null; }
}

// Registry of smoothers per DOM element id. Callers use setSmoothed('hrL', 72)
// and the rAF loop writes the eased number to the DOM once per VSYNC.
const _smoothed = new Map();
function _smootherFor(id){
    if(!_smoothed.has(id)){
        // Defaults are value-appropriate — HR has dead-zone 0.5 BPM, SpO2 has 0.5%, etc.
        const defaults = {
            hrL:   {ms:60, dead:0.5,  precision:0},
            hrC:   {ms:60, dead:0.5,  precision:0},
            oxiHR: {ms:60, dead:0.5,  precision:0},
            oxiSp: {ms:60, dead:0.5,  precision:0},
            rHR:   {ms:60, dead:0.5,  precision:0},
            rBR:   {ms:80, dead:0.5,  precision:0},
            rHR2:  {ms:60, dead:0.5,  precision:0},
            rBR2:  {ms:80, dead:0.5,  precision:0},
            rDist: {ms:80, dead:0.02, precision:2},
            rDist2:{ms:80, dead:0.02, precision:2},
            vcF0:  {ms:60, dead:1.0,  precision:0},
        };
        _smoothed.set(id, new SmoothedNumber(defaults[id] || {ms:60, dead:0, precision:0}));
    }
    return _smoothed.get(id);
}
function setSmoothed(id, value, colour){
    const sm = _smootherFor(id);
    sm.set(value);
    // Apply colour immediately — only the number animates, not the colour.
    if(colour){ const el = document.getElementById(id); if(el) el.style.color = colour; }
}
function _flushSmoothed(){
    for(const [id, sm] of _smoothed){
        const out = sm.render();
        if(out == null) continue;
        const el = document.getElementById(id);
        if(!el) continue;
        if(el.textContent !== String(out)) el.textContent = out;
    }
}

// ─── Step 22: batched DOM writes scheduler ───────────────────────────
// Queues up DOM mutations from anywhere in the codebase and flushes
// them once per rAF tick. Prevents multiple forced-layouts per frame.
const _domQueue = new Map();
function sElQueued(id, text, colour){
    _domQueue.set(id, {text, colour});
}
function _flushDomQueue(){
    if(_domQueue.size === 0) return;
    for(const [id, {text, colour}] of _domQueue){
        const el = document.getElementById(id);
        if(!el) continue;
        if(text !== undefined && el.textContent !== String(text)) el.textContent = String(text);
        if(colour) el.style.color = colour;
    }
    _domQueue.clear();
}

// ─── Steps 17-18: face mask canvas — auto-upgrade to WebGL2 if available ─
// Keeps the 2D canvas as a fallback if WebGL2 isn't supported in this
// browser. On WebGL2 we batch all 5 faces × 478 landmarks into a single
// gl.drawArrays(POINTS) call, which the GPU resolves in <0.5 ms.
let _webgl = null;
let _webglProg = null;
let _webglVbo = null;
function _initWebGL(){
    try {
        const cv = document.getElementById('overlay');
        if(!cv) return false;
        const gl = cv.getContext('webgl2', {premultipliedAlpha: true, antialias: true});
        if(!gl) return false;
        const vs = `#version 300 es
            in vec2 a_pos; in vec3 a_col;
            out vec3 v_col;
            uniform vec2 u_size;
            void main(){
                // convert normalized [0,1] canvas coords → NDC, Y flipped.
                vec2 ndc = vec2(a_pos.x * 2.0 - 1.0, 1.0 - a_pos.y * 2.0);
                gl_Position = vec4(ndc, 0.0, 1.0);
                gl_PointSize = 1.4;
                v_col = a_col;
            }`;
        const fs = `#version 300 es
            precision mediump float;
            in vec3 v_col;
            out vec4 outCol;
            void main(){ outCol = vec4(v_col, 0.65); }`;
        function compile(type, src){
            const s = gl.createShader(type);
            gl.shaderSource(s, src); gl.compileShader(s);
            if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)){
                console.warn('[WebGL] shader compile', gl.getShaderInfoLog(s));
                return null;
            }
            return s;
        }
        const v = compile(gl.VERTEX_SHADER, vs);
        const f = compile(gl.FRAGMENT_SHADER, fs);
        if(!v || !f) return false;
        const p = gl.createProgram();
        gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
        if(!gl.getProgramParameter(p, gl.LINK_STATUS)){
            console.warn('[WebGL] link', gl.getProgramInfoLog(p));
            return false;
        }
        gl.useProgram(p);
        _webgl = gl;
        _webglProg = p;
        _webglVbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, _webglVbo);
        const aPos = gl.getAttribLocation(p, 'a_pos');
        const aCol = gl.getAttribLocation(p, 'a_col');
        gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aCol);
        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 20, 0);
        gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 20, 8);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        console.log('[WebGL] face-mask renderer ready');
        return true;
    } catch(e){
        console.warn('[WebGL] init failed:', e);
        return false;
    }
}
function _drawWebGL(faces){
    if(!_webgl) return false;
    const gl = _webgl;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if(!faces || !faces.length) return true;
    // Flatten: per landmark → [x, y, r, g, b]
    let totalPts = 0;
    for(const f of faces){ if(f.lm) totalPts += f.lm.length; }
    if(totalPts === 0) return true;
    const buf = new Float32Array(totalPts * 5);
    let k = 0;
    for(const f of faces){
        const df = f.df || {};
        const dm = (_latestWS && _latestWS.av && _latestWS.av.dominant_emotion)
                   || df.emotion?.dominant || 'neutral';
        const hex = (ec && ec[dm]) || '#22c55e';
        const r = parseInt(hex.slice(1,3),16)/255;
        const g = parseInt(hex.slice(3,5),16)/255;
        const b = parseInt(hex.slice(5,7),16)/255;
        for(const lm of (f.lm || [])){
            const lx = Array.isArray(lm) ? lm[0] : lm.x;
            const ly = Array.isArray(lm) ? lm[1] : lm.y;
            buf[k++] = lx; buf[k++] = ly; buf[k++] = r; buf[k++] = g; buf[k++] = b;
        }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, _webglVbo);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.POINTS, 0, totalPts);
    return true;
}

// ─── Step 23: content-visibility on off-screen panels ────────────────
// Applied via CSS (see the stylesheet edit further down). This helper
// lets Chrome skip layout/paint for panels that aren't in the viewport.
function _applyContentVisibility(){
    // Panels we mark as 'auto' — they self-manage render cost when collapsed.
    ['aiHistBody', 'faceCards'].forEach(id => {
        const el = document.getElementById(id);
        if(el) el.style.contentVisibility = 'auto';
    });
}

// v9.1.6 + v9.2: continuous draw loop — everything flows through here.
let _latestWS = null;
let _latestWSArrivedAt = 0;
let _drawLoopErrorBurst = 0;
let _webglReady = false;
let _panelTick = 0;   // v9.5.x — rAF throttle for heavy panel updates
function _drawLoop(){
    if(!on) return;
    _pmark('draw-start');
    // Step 17/18 — try WebGL2 once on first paint
    if(_webglReady === false){ _webglReady = _initWebGL() ? true : 'fallback'; _applyContentVisibility(); }
    if(_latestWS){
        try {
            // v9.5.x — INSTANT TRACKING: when client-side MediaPipe is live
            // the face mesh comes from `instant.faces` (computed locally at
            // camera-rVFC cadence, zero network RTT). Fall back to the
            // WS-driven predicted path otherwise.
            let payloadForDraw;
            if(FF.CLIENT_TRACKER && _clientTracker &&
               instant.faces && instant.faces.length){
                // v9.5.x — lift the dominant emotion from the first face
                // into a shadow av so uEmo / draw() can read it live.
                const _f0 = instant.faces[0];
                const _shadowAv = Object.assign({}, (_latestWS.av || {}));
                if(_f0 && _f0._dominant_emotion){
                    _shadowAv.dominant_emotion = _f0._dominant_emotion;
                }
                if(_f0 && _f0.iris){
                    _shadowAv.pupil_dilation = _f0.iris.avg_dilation;
                }
                payloadForDraw = Object.assign({}, _latestWS, {
                    faces: instant.faces,
                    av: _shadowAv,
                });
            } else {
                // Step 19 — predictive interpolation between server frames
                const predicted = _predictedFaces(performance.now());
                payloadForDraw = predicted
                    ? Object.assign({}, _latestWS, {faces: predicted})
                    : _latestWS;
            }
            // v9.3 step 3 — three-tier overlay renderer: WebGPU → WebGL2 → 2D.
            // Each tier draws only the landmark points; iris rings, gaze
            // arrows, and head-pose annotations live on the 2D canvas so
            // they survive regardless of which tier won.
            let _renderedByGPU = false;
            if(FF.WEBGPU && _gpuReady){
                _renderedByGPU = _drawWebGPU(payloadForDraw.faces);
            }
            let _glOk = false;
            if(!_renderedByGPU){
                _glOk = _webglReady === true && _drawWebGL(payloadForDraw.faces);
            }
            // Skip the 2D landmark loop when any GPU tier already painted them.
            // Saves 5-15 ms / frame with 478 landmarks per face.
            const _skipLandmarks = _renderedByGPU || _glOk;
            draw(payloadForDraw, _skipLandmarks);
            // v9.5.x — throttle heavy panel updates to ~15 Hz instead of
            // rAF-rate (60 Hz). The underlying data only changes at 2-10 Hz,
            // so rebuilding emotion bars / HR cards every vsync is waste.
            // v9.5.x — when client-tracker is driving, make the panels
            // instant (camera rVFC) for emotion/blink/pupil. The heavy HR
            // + AV panels stay at 15 Hz because their source data only
            // changes at a few Hz.
            const _clientDriving = !!(FF.CLIENT_TRACKER && _clientTracker &&
                                      instant.faces && instant.faces.length);
            const _panelDue = !_panelTick || (performance.now() - _panelTick) > 66;
            if(_clientDriving){
                // Emotion + per-face card rebuild use client-computed
                // values directly — update every rAF for instant feel.
                uEmo(payloadForDraw.faces, payloadForDraw);
            } else if(_panelDue){
                uEmo(_latestWS.faces, _latestWS);
            }
            if(_panelDue){
                _panelTick = performance.now();
                uHR(_latestWS);
                if(payloadForDraw.av){
                    uAV(payloadForDraw.av, _latestWS.ref);
                    updateHRDelta(payloadForDraw.av);
                }
            }
            _drawLoopErrorBurst = 0;
        } catch(e){
            if((++_drawLoopErrorBurst) % 60 === 0){
                console.error('[drawLoop] sustained error:', e);
            }
        }
    }
    _flushSmoothed();   // Step 25/30 — write eased numbers to DOM
    _flushDomQueue();   // Step 22 — batched DOM writes
    _pmark('draw-end');
    _pmeasure('frame-draw', 'draw-start', 'draw-end');
    requestAnimationFrame(_drawLoop);
}
function lp(){
    if(!_sessionStart) _sessionStart = Date.now();
    // Kick off the continuous draw loop once
    requestAnimationFrame(_drawLoop);
    function _tick(){
        if(!on) return;
        if(document.hidden){
            // Throttled tick when tab is hidden so we don't burn cycles
            const now=Date.now();
            if(now-_lastHiddenSend >= 1000){_lastHiddenSend=now;_captureAndSend();}
            setTimeout(_tick, 500);
            return;
        }
        _captureAndSend();
        requestAnimationFrame(_tick);
    }
    _tick();
}
// HTTP fallback if WebSocket fails
async function lpHTTP(){
    while(on){
        if(inflight<MAX_INFLIGHT){
            inflight++;
            const tmp=document.createElement('canvas');tmp.width=W;tmp.height=H;
            tmp.getContext('2d').drawImage(v,0,0,W,H);
            tmp.toBlob(blob=>{
                const fd=new FormData();fd.append('file',blob,'f.jpg');
                const ep=(n%2===0)?'/detect/full':'/detect/fast';
                const t0=performance.now();
                fetch(ep,{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
                    const ms=Math.round(performance.now()-t0);fpsArr.push(ms);if(fpsArr.length>30)fpsArr.shift();
                    const avgMs=Math.round(fpsArr.reduce((a,b)=>a+b,0)/fpsArr.length);const fps=Math.round(1000/avgMs);
                    draw(d);uEmo(d.faces,d);uHR(d);if(d.av)uAV(d.av,d.ref);if(d.ai&&d.ai.length)uAI(d.ai);
                    // v9.7.5: PAIRS badge fix — was reading `d.faces.length` (number
        // of faces in frame), which is why it permanently showed 1 for
        // a single-face setup regardless of how many QLoRA training
        // pairs had actually been logged. The real count now comes
        // from /qlora/status polled at 1 Hz (see _qloraBadgeRefresh).
        // We leave this assignment as a no-op to avoid churning the
        // badge between WS frames; the poll-based refresh below owns
        // the value.
                    document.getElementById('stats').textContent=fps+' fps | '+avgMs+'ms | '+d.mode;
                    inflight--;}).catch(e=>{document.getElementById('stats').textContent='Err: '+e.message;inflight--;});
                n++;
            },'image/jpeg',0.45);
        }
        await new Promise(r=>setTimeout(r,1));
    }
}
// ══════════════════════════════════════════
//  TRUTH TEST — Multi-modal deception analysis
// ══════════════════════════════════════════
let truthState='idle'; // idle|baseline|question|recording|analyzing
let truthMR=null; // MediaRecorder
let truthChunks=[];
let truthQuestion='';
let baselineTimer=null;
let answerFeedInterval=null;

async function truthTest(){
    if(truthState!=='idle'){truthReset();return;}
    // Step 1: Start baseline collection
    truthState='baseline';
    document.getElementById('truthBtn').textContent='CANCEL';
    document.getElementById('truthBtn').style.background='#71717a';
    document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444;font-size:11px">TRUTH TEST: Look at camera calmly. Collecting baseline...</div><div style="text-align:center;margin-top:4px"><div id="truthProgress" style="font-family:monospace;font-size:20px;color:#ef4444">20</div><div class="gl">seconds remaining</div></div>';
    await fetch('/truth/start_baseline',{method:'POST'});
    let countdown=20;
    baselineTimer=setInterval(async()=>{
        countdown--;
        const el=document.getElementById('truthProgress');
        if(el)el.textContent=countdown;
        await fetch('/truth/feed_baseline',{method:'POST'});
        if(countdown<=0){
            clearInterval(baselineTimer);
            truthAskQuestion();
        }
    },1000);
}

function truthAskQuestion(){
    truthState='question';
    document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#f59e0b;font-size:11px">TRUTH TEST: Type your question, then click RECORD ANSWER</div><div style="padding:4px"><input id="truthQ" placeholder="Type question here..." style="width:100%;padding:6px;font-size:11px;background:#18181b;color:#fafafa;border:1px solid #27272a;border-radius:4px;margin-bottom:4px"><button onclick="truthStartRecording()" style="width:100%;padding:6px;font-size:11px;font-weight:800;background:#ef4444;color:#fff;border:2px solid #dc2626;border-radius:4px;cursor:pointer">RECORD ANSWER</button></div>';
}

async function truthStartRecording(){
    const q=document.getElementById('truthQ')?.value||'';
    truthQuestion=q;
    truthState='recording';
    // Start answer biometric capture on server
    await fetch('/truth/start_answer',{method:'POST'});
    // Start mic recording
    try{
        const stream=await navigator.mediaDevices.getUserMedia({audio:true});
        truthMR=new MediaRecorder(stream,{mimeType:'audio/webm;codecs=opus'});
        truthChunks=[];
        truthMR.ondataavailable=e=>{if(e.data.size>0)truthChunks.push(e.data);};
        truthMR.onstop=()=>{stream.getTracks().forEach(t=>t.stop());truthAnalyze();};
        truthMR.start();
        // Feed answer biometrics every 500ms
        answerFeedInterval=setInterval(()=>{fetch('/truth/feed_answer',{method:'POST'});},500);
        document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444;font-size:11px">RECORDING... Subject should answer now</div><div style="text-align:center;margin:8px"><div style="width:12px;height:12px;border-radius:50%;background:#ef4444;margin:0 auto;animation:pulse 0.8s infinite"></div><div class="gl" style="margin-top:4px">Recording audio + biometrics</div><div style="margin-top:8px"><button onclick="truthStopRecording()" style="padding:8px 24px;font-size:12px;font-weight:800;background:#f59e0b;color:#000;border:2px solid #d97706;border-radius:4px;cursor:pointer">STOP RECORDING</button></div></div>';
    }catch(e){
        document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">Mic error: '+e.message+'</div>';
        truthReset();
    }
}

function truthStopRecording(){
    if(answerFeedInterval){clearInterval(answerFeedInterval);answerFeedInterval=null;}
    if(truthMR&&truthMR.state==='recording')truthMR.stop();
}

async function truthAnalyze(){
    truthState='analyzing';
    document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#a855f7;font-size:11px">Analyzing audio + biometrics + vocal stress...</div><div style="text-align:center;margin:4px"><div id="aiSpinner" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#a855f7;animation:pulse 0.6s infinite"></div></div>';
    try{
        const audioBlob=new Blob(truthChunks,{type:'audio/webm'});
        const fd=new FormData();
        fd.append('file',audioBlob,'answer.webm');
        fd.append('question',truthQuestion);
        const r=await fetch('/truth/analyze',{method:'POST',body:fd});
        const d=await r.json();
        // Display results
        let html='<div style="border-left:3px solid #ef4444;padding:6px;margin-bottom:4px;background:#18181b;border-radius:0 4px 4px 0">';
        html+='<div style="font-size:7px;color:#52525b;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px">Truth Test Result</div>';
        if(d.transcript?.text)html+='<div style="font-size:12px;color:#a1a1aa;margin-bottom:4px">Transcript: "'+d.transcript.text+'"</div>';
        const ind=d.deception_indicators||{};
        const score=ind.composite_score||0;
        const sc=score>0.6?'#ef4444':score>0.4?'#f59e0b':score>0.2?'#3b82f6':'#22c55e';
        html+='<div style="font-family:monospace;font-size:18px;font-weight:800;color:'+sc+'" title="Composite arousal / cognitive-load score from multimodal physiological signals. NOT a binary deception verdict — DePaulo et al. 2003 meta-analysis shows individual cues have small effect sizes; treat as a research indicator only.">'+Math.round(score*100)+'% STRESS COMPOSITE</div>';
        html+='<div style="font-size:9px;font-weight:700;color:'+sc+';margin-bottom:4px">'+(ind.verdict||'')+'</div>';
        // Top indicators
        const inds=ind.indicators||{};
        const sorted=Object.entries(inds).sort((a,b)=>(b[1].score||0)-(a[1].score||0)).slice(0,5);
        sorted.forEach(([k,v])=>{
            const isc=v.score>0.5?'#ef4444':v.score>0.25?'#f59e0b':'#22c55e';
            html+='<div class="br"><div class="br-t"><span>'+v.label+'</span><span style="color:'+isc+'">'+Math.round(v.score*100)+'%</span></div><div class="br-k"><div class="br-f" style="width:'+v.score*100+'%;background:'+isc+'"></div></div></div>';
        });
        html+='</div>';
        // LLM analysis
        if(d.llm_analysis)html+='<div class="ai-msg"><div class="ai-t">'+new Date().toLocaleTimeString()+' TRUTH ANALYSIS</div><div class="ai-x">'+d.llm_analysis+'</div></div>';
        _cache['ai']='';
        document.getElementById('aiBox').innerHTML=html;
        document.getElementById('aiTime').textContent='Truth Test: '+new Date().toLocaleTimeString();
    }catch(e){
        document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">Analysis error: '+e.message+'</div>';
    }
    truthReset(true);
}

function truthReset(keepResults){
    truthState='idle';
    if(baselineTimer){clearInterval(baselineTimer);baselineTimer=null;}
    if(answerFeedInterval){clearInterval(answerFeedInterval);answerFeedInterval=null;}
    if(truthMR&&truthMR.state==='recording')truthMR.stop();
    truthMR=null;truthChunks=[];
    document.getElementById('truthBtn').textContent='TRUTH TEST';
    document.getElementById('truthBtn').style.background='#ef4444';
    if(!keepResults){
        document.getElementById('aiBox').innerHTML='<div class="ai-e">Click DODO or enable MON</div>';
        fetch('/truth/reset',{method:'POST'});
    }
}
// ══════════════════════════════════════════
//  CONTINUOUS VOICE MONITOR
// ══════════════════════════════════════════
let micOn=false;let micStream=null;let micRecorder=null;let micInterval=null;
let micAudioCtx=null;let micAnalyser=null;let micLevelRAF=null;
// v7.9: watchdog handle — resumes a suspended AudioContext so the meter
// never stays stuck at 0 when the tab is backgrounded / OS sleeps briefly.
let micHealWatchdog=null;

// Page-load boot routine.
// v8.7 FINAL FIX: no more auto-start. Modern browsers (Chrome, Edge, Firefox)
// increasingly defer or block getUserMedia() calls that happen without a
// user gesture, and the silent failure + half-initialized state was the
// ACTUAL root cause of the "START does nothing" symptom. Users must click
// START once; the button is styled prominently and the status line makes
// progress obvious. The /status fetch remains so the governor / faces /
// LLM badges are populated immediately.
(async function(){
    const dbg=document.getElementById('stats');
    dbg.textContent='Ready — click START to enable camera';dbg.style.color='#22c55e';
    const sw=document.getElementById('sWcam');
    if(sw){sw.textContent='CAM: click START';sw.style.color='#f59e0b';}
    fetch('/status').then(r=>r.ok?r.json():null).then(s=>{
        if(s){console.log('[BOOT] Server status OK:',s);uGovBudget(s);}
    }).catch(e=>{console.warn('[BOOT] /status fetch failed (non-fatal):',e.message);});
})();

async function toggleMic(){
    micOn=!micOn;
    const b=document.getElementById('micBtn');
    if(micOn){
        try{
            micStream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});
            await fetch('/voice/start',{method:'POST'});
            b.textContent='MIC: ON';b.style.background='#22c55e';b.style.borderColor='#16a34a';
            micAudioCtx=new AudioContext({sampleRate:16000});
            // v7.9: robust resume. Chrome sometimes leaves the context in
            // 'suspended' even after a user gesture (e.g. page reload while
            // mic permission was remembered). Poll-retry up to 5 × 50 ms
            // so the meter is guaranteed to start.
            for(let i=0;i<5 && micAudioCtx.state!=='running';i++){
                try{ await micAudioCtx.resume(); }catch(_){}
                if(micAudioCtx.state==='running') break;
                await new Promise(r=>setTimeout(r,50));
            }
            const src=micAudioCtx.createMediaStreamSource(micStream);
            micAnalyser=micAudioCtx.createAnalyser();
            micAnalyser.fftSize=2048;micAnalyser.smoothingTimeConstant=0.8;
            src.connect(micAnalyser);
            // v7.9: self-heal watchdog. If the AudioContext suspends later
            // (tab backgrounded, OS power event, policy change), resume it
            // automatically — the meter never gets stuck at 0 silently.
            if(micHealWatchdog){ clearInterval(micHealWatchdog); micHealWatchdog=null; }
            micHealWatchdog=setInterval(()=>{
                if(!micOn||!micAudioCtx){ clearInterval(micHealWatchdog); micHealWatchdog=null; return; }
                if(micAudioCtx.state==='suspended'){ micAudioCtx.resume().catch(()=>{}); }
                // If the RAF loop somehow died (e.g. an exception or tab
                // throttling), restart it on the next frame.
                if(micOn && !micLevelRAF && typeof updateMicLevel==='function'){
                    micLevelRAF=requestAnimationFrame(updateMicLevel);
                }
            },1000);
            // DO NOT connect to destination (would play mic through speakers)
            // v7.7: proper time-domain RMS meter (dBFS). Independent of VAD
            // so the bar always reflects true mic level (fixes stuck meter).
            const levelBuf=new Float32Array(micAnalyser.fftSize);
            let _peakDb=-60, _peakHoldMs=0;
            function updateMicLevel(){
                if(!micOn||!micAnalyser){ micLevelRAF=null; return; }
                micAnalyser.getFloatTimeDomainData(levelBuf);
                let sum=0;for(let i=0;i<levelBuf.length;i++){sum+=levelBuf[i]*levelBuf[i];}
                const rms=Math.sqrt(sum/levelBuf.length);
                // dBFS: 0 dB = full-scale. Clamp to [-60, 0].
                let db=rms>1e-6?20*Math.log10(rms):-60;
                if(db<-60)db=-60;if(db>0)db=0;
                // Peak hold for 600ms so transients are visible.
                const now=performance.now();
                if(db>_peakDb){_peakDb=db;_peakHoldMs=now;}
                else if(now-_peakHoldMs>600){_peakDb=Math.max(db,_peakDb-0.5);}
                const pct=Math.max(0,Math.min(100,(db+60)/60*100));
                const bar=document.getElementById('micBar');
                if(bar)bar.style.width=pct+'%';
                sEl('micDb',db.toFixed(0)+' dBFS');
                // Color flip: amber when clipping (>-6 dB)
                const hi=document.getElementById('micPeak');
                if(hi){hi.style.left=Math.max(0,Math.min(100,(_peakDb+60)/60*100))+'%';}
                micLevelRAF=requestAnimationFrame(updateMicLevel);
            }
            // v7.9: kick RAF immediately AND schedule a 120 ms fallback
            // start — guards against the first RAF being throttled by a
            // hidden tab, page-load reflow, or a suspended AudioContext
            // that only resumes after the initial resume() race.
            updateMicLevel();
            setTimeout(()=>{ if(micOn && !micLevelRAF) updateMicLevel(); },120);
            // PCM capture: buffer 3s of time-domain data, then send
            const pcmSize=1024;
            const timeBuf=new Float32Array(pcmSize);
            let pcmAccum=[];
            let pcmTotal=0;
            const targetSamples=micAudioCtx.sampleRate*3;
            micInterval=setInterval(()=>{
                if(!micOn||!micAnalyser)return;
                micAnalyser.getFloatTimeDomainData(timeBuf);
                pcmAccum.push(new Float32Array(timeBuf));
                pcmTotal+=pcmSize;
                if(pcmTotal>=targetSamples){
                    const full=new Float32Array(pcmTotal);
                    let off=0;for(const b of pcmAccum){full.set(b,off);off+=b.length;}
                    pcmAccum=[];pcmTotal=0;
                    const sr=micAudioCtx.sampleRate;
                    const payload=new Float32Array(1+full.length);
                    payload[0]=sr;payload.set(full,1);
                    fetch('/voice/pcm',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:payload.buffer}).then(r=>r.json()).then(d=>{
                        if(d&&!d.error){
                            // Update voice metrics display directly from response
                            if(d.f0_hz)sEl('vcF0',Math.round(d.f0_hz)+'');
                            if(d.jitter!=null)sEl('vcJit',d.jitter.toFixed(3));
                            const sl=d.stress_level||'--';
                            sEl('vcStr',sl.toUpperCase(),sl==='high'?'#ef4444':sl==='elevated'?'#f59e0b':'#22c55e');
                        }
                    }).catch(()=>{});
                }
            },Math.round(pcmSize/micAudioCtx.sampleRate*1000));
        }catch(e){micOn=false;b.textContent='MIC: ERR ('+e.message+')';b.style.background='#ef4444';console.error('Mic error:',e);}
    }else{
        if(micInterval){clearInterval(micInterval);micInterval=null;}
        if(micLevelRAF){cancelAnimationFrame(micLevelRAF);micLevelRAF=null;}
        if(micHealWatchdog){clearInterval(micHealWatchdog);micHealWatchdog=null;}
        if(micAudioCtx){micAudioCtx.close().catch(()=>{});micAudioCtx=null;micAnalyser=null;}
        if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null;}
        await fetch('/voice/stop',{method:'POST'}).catch(()=>{});
        b.textContent='MIC: OFF';b.style.background='#71717a';b.style.borderColor='#52525b';
        document.getElementById('micBar').style.width='0%';sEl('micDb','-- dB');
        sEl('vcF0','--');sEl('vcJit','--');sEl('vcStr','--');
    }
}

// Legacy functions kept for compatibility (truth test uses MediaRecorder separately)
function startMicChunks(){}
function stopMicChunks(){
    if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null;}
}
async function radarReconnect(){
    sEl('sRadar','RADAR: reconnecting…','#f59e0b');
    try{
        const r = await fetch('/radar/reconnect',{method:'POST'});
        const j = await r.json();
        if(j.ok){sEl('sRadar','RADAR: '+(j.status?.port||'OK'),'#22c55e');}
        else{sEl('sRadar','RADAR: '+(j.status?.error||'failed'),'#ef4444');}
    }catch(e){sEl('sRadar','RADAR: '+e.message,'#ef4444');}
}
async function forensicExport(){
    const pid = prompt('person_id to export (empty = whole session):','')||'';
    const url = '/forensic/export?download=true'+(pid?('&person_id='+encodeURIComponent(pid)):'');
    window.open(url,'_blank');
}
// v8.3: Pleth waveform mini-canvas (50 sample ring buffer)
const _plethBuf=new Array(50).fill(0);let _plethIdx=0;
function drawPleth(val){
    _plethBuf[_plethIdx%50]=val||0;_plethIdx++;
    const cv=document.getElementById('plethCanvas');if(!cv)return;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,120,30);
    ctx.strokeStyle='#22c55e';ctx.lineWidth=1;ctx.beginPath();
    for(let i=0;i<50;i++){
        const idx=(_plethIdx+i)%50;
        const x=i*(120/49);const y=30-(_plethBuf[idx]/512)*28;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();
}
// v8.4: SpO2 sparkline (30-point circular buffer)
const _spo2Buf=new Array(30).fill(0);let _spo2Idx=0;
function drawSpo2Spark(val){
    if(val==null)return;
    _spo2Buf[_spo2Idx%30]=val;_spo2Idx++;
    const cv=document.getElementById('spo2Spark');if(!cv)return;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,60,16);
    // Auto-scale: find min/max in buffer
    let mn=100,mx=90;for(let i=0;i<30;i++){const v=_spo2Buf[i];if(v>0){if(v<mn)mn=v;if(v>mx)mx=v;}}
    if(mx-mn<2){mn-=1;mx+=1;}
    ctx.strokeStyle='#06b6d4';ctx.lineWidth=1;ctx.beginPath();
    for(let i=0;i<30;i++){const idx=(_spo2Idx+i)%30;const x=i*(60/29);const y=16-(((_spo2Buf[idx]||mn)-mn)/(mx-mn))*14;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.stroke();
}
// v8.4: Pleth waveform auto-scale (replaces fixed /512 scale)
const _plethBufV2=new Array(50).fill(0);let _plethIdxV2=0;
function drawPlethAutoScale(val){
    _plethBufV2[_plethIdxV2%50]=val||0;_plethIdxV2++;
    const cv=document.getElementById('plethCanvas');if(!cv)return;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,120,30);
    let mn=9999,mx=-9999;for(let i=0;i<50;i++){const v=_plethBufV2[i];if(v<mn)mn=v;if(v>mx)mx=v;}
    if(mx-mn<1){mn-=1;mx+=1;}
    ctx.strokeStyle='#22c55e';ctx.lineWidth=1;ctx.beginPath();
    for(let i=0;i<50;i++){const idx=(_plethIdxV2+i)%50;const x=i*(120/49);const y=30-(((_plethBufV2[idx]||mn)-mn)/(mx-mn))*28;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.stroke();
}
// v8.4: deception trend sparkline (60 points per person)
const _decBuf=new Array(60).fill(0);let _decIdx=0;
function drawDecSpark(score){
    _decBuf[_decIdx%60]=score||0;_decIdx++;
}
// v8.4: oximeter status dot polling
setInterval(async()=>{
    try{const r=await fetch('/oximeter/status');const d=await r.json();
    const dot=document.getElementById('sOxiDot');if(!dot)return;
    const modeTag=d.mode?(' ['+d.mode.toUpperCase()+']'):'';
    if(d.connected&&d.data){dot.style.background='#22c55e';dot.title='Oximeter: connected'+modeTag+' '+d.port;}
    else if(d.connected){dot.style.background='#f59e0b';dot.title='Oximeter: connected'+modeTag+', waiting for finger';}
    else if(d.error){dot.style.background='#f59e0b';dot.title='Oximeter: '+d.error;}
    else{dot.style.background='#52525b';dot.title='Oximeter: searching (HID/serial)';}
    }catch(e){}
},3000);
// v8.4: HR delta display (radar HR vs rPPG HR)
function updateHRDelta(av){
    const el=document.getElementById('sDelta');if(!el)return;
    if(av.radar_hr&&av.heart_rate_bpm){
        const delta=Math.round(av.radar_hr-av.heart_rate_bpm);
        const abs_d=Math.abs(delta);
        const col=abs_d>10?'#ef4444':abs_d>5?'#f59e0b':'#22c55e';
        el.textContent='HR delta:'+(delta>=0?'+':'')+delta;el.style.color=col;
    }else{el.textContent='';}
}
// v8.4: AI history buffer (last 5)
const _aiHist=[];
function pushAIHistory(text){
    _aiHist.push({t:new Date().toLocaleTimeString(),text:text.substring(0,300)});
    if(_aiHist.length>5)_aiHist.shift();
    const el=document.getElementById('aiHistBody');if(!el)return;
    el.innerHTML=_aiHist.map(h=>'<div style="border-bottom:1px solid #27272a;padding:2px 0"><b>'+h.t+'</b> '+h.text+'</div>').join('');
}
// v8.4: room analyze button
async function roomAnalyze(){
    document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#7c3aed">Running room analysis...</div>';
    try{const r=await fetch('/analyze/persons',{method:'POST'});const d=await r.json();
    if(d.room){_cache['ai']='';uAI([{t:new Date().toLocaleTimeString(),text:'[ROOM] '+d.room}]);pushAIHistory('[ROOM] '+d.room);}
    else if(d.error){document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">'+d.error+'</div>';}
    }catch(e){document.getElementById('aiBox').innerHTML='<div class="ai-e" style="color:#ef4444">'+e.message+'</div>';}
}
// v8.4: dark/light theme toggle
let _darkMode=localStorage.getItem('dodo_theme')!=='light';
function toggleTheme(){
    _darkMode=!_darkMode;
    localStorage.setItem('dodo_theme',_darkMode?'dark':'light');
    applyTheme();
}
function applyTheme(){
    const b=document.body;
    if(_darkMode){b.style.background='#09090b';b.style.color='#fafafa';document.getElementById('themeBtn').innerHTML='&#9790;';}
    else{b.style.background='#f8f9fa';b.style.color='#212529';document.getElementById('themeBtn').innerHTML='&#9788;';}
}
applyTheme();
// v8.4: keyboard shortcuts
document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
    if(e.key==='m'||e.key==='M'){toggleMon();e.preventDefault();}
    if(e.key==='r'||e.key==='R'){rs();e.preventDefault();}
    if(e.key==='a'||e.key==='A'){askAI();e.preventDefault();}
    // v8.7 FINAL FIX: spacebar only STOPS the camera (keeps the emergency
    // kill-switch) but never STARTS it — go() is driven exclusively by the
    // visible START button, so there is zero chance of a race with keybinds.
    if(e.key===' ' && on){stop();e.preventDefault();}
    // v9.8: Ctrl+Alt+R hard-resets all per-subject calibrations on the
    // server (patch_depth EMA baseline, blink detector face-rebind state,
    // any face-keyed Platt offsets). Useful when you swap subject mid-
    // session and don't want their stale baseline blocking patch_depth
    // from speaking up again. NOT bound to plain 'R' to avoid clashing
    // with rs() (rPPG reset).
    if((e.ctrlKey || e.metaKey) && e.altKey && (e.key==='r' || e.key==='R')){
        e.preventDefault();
        fetch('/calibration/reset_all',{method:'POST'}).then(r=>r.json()).then(j=>{
            try{window.__diag('calibration reset: '+JSON.stringify(j),'#22c55e');}catch(_){}
        }).catch(err=>{
            try{window.__diag('calibration reset failed: '+err.message,'#ef4444');}catch(_){}
        });
    }
});
// v8.4: WebSocket exponential backoff with jitter
let _wsBackoff=1000;
// v8.3: QLoRA export button
async function exportQLoRA(){
    const btn=document.getElementById('qloraBtn');
    btn.textContent='EXPORTING...';btn.style.color='#f59e0b';
    try{
        const r=await fetch('/qlora/export?session_id=latest&consent=true',{method:'POST'});
        const d=await r.json();
        if(d.error){btn.textContent='ERROR';btn.style.color='#ef4444';setTimeout(()=>{btn.textContent='EXPORT QLoRA';btn.style.color='#a78bfa';},3000);}
        else{btn.textContent='EXPORTED ('+d.pairs+')';btn.style.color='#22c55e';setTimeout(()=>{btn.textContent='EXPORT QLoRA';btn.style.color='#a78bfa';},5000);}
    }catch(e){btn.textContent='FAILED';btn.style.color='#ef4444';setTimeout(()=>{btn.textContent='EXPORT QLoRA';btn.style.color='#a78bfa';},3000);}
}

