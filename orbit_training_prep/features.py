from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any
import numpy as np
from .geometry_bridge import make_geometry
from .schema import NOOP_TARGET_SLOT, P_MAX, relative_owner, safe_float
from .viability import _is_native_fleet_observation, _native_fleet_target_and_eta
CENTER=50.0; BOARD=100.0; ROTATION_RADIUS_LIMIT=50.0; SHIP_LOG_DENOM=math.log1p(1000.0)
PLANET_FEATURE_NAMES=["alive","rel_owner_neutral","rel_owner_own","rel_owner_enemy","x_centered","y_centered","radius_norm","ships_log_norm","production_norm","is_comet","is_orbiting","distance_center_norm","owner_ship_share","owner_prod_share","projected_garrison_20","under_threat_20"]
GLOBAL_FEATURE_NAMES=["step_norm","remaining_steps_norm","is_2p","is_4p","my_ship_share","my_prod_share","my_planet_share","leader_ship_gap_norm","leader_prod_gap_norm","weakest_enemy_ship_gap_norm"]
TARGET_STATE_FEATURE_NAMES=["nearest_own_eta_to_target","nearest_enemy_eta_to_target","enemy_before_own_flag","hostile_arrivals_before_10","projected_owner_20","projected_garrison_20","target_contested_flag","target_easy_neutral_flag","target_high_prod_flag"]
PAIR_FEATURE_NAMES=["capture_needed","capture_ratio","surplus_after_capture","roi_prod_per_ship","is_neutral","is_enemy","is_own","cheap_neutral","high_prod_target","distance","angle_sin","angle_cos","geom_viable_any_amount","geom_viable_amount_frac","geom_no_viable_amount_flag","safe_sendable_ships","post_send_frac_capture","overkill_ratio_capture","enemy_before_us","our_arrival_margin","enemy_can_capture_before_us","local_ship_advantage_20","projected_garrison_at_arrival","projected_owner_at_arrival","is_noop_candidate"]
@dataclass(frozen=True)
class FeatureState:
    planet_features:np.ndarray; global_features:np.ndarray; target_state_features:np.ndarray
def ships_from_log_norm(x:float)->float: return math.expm1(float(x)*SHIP_LOG_DENOM)
def _step_norm(obs:dict[str,Any])->float: return safe_float(obs.get("step"),0.0)/max(safe_float(obs.get("episode_steps"),500.0),1.0)
def _num_players(obs:dict[str,Any])->int:
    owners={int(p[1]) for p in obs.get("planets",[]) if len(p)>=7 and int(p[1])>=0}; pc=int(obs.get("num_players",obs.get("players",0)) or 0); return max(pc,len(owners),2)
def _owner_totals(obs:dict[str,Any])->dict[int,dict[str,float]]:
    totals={}
    for p in obs.get("planets",[])[:P_MAX]:
        if len(p)<7 or int(p[1])<0: continue
        r=totals.setdefault(int(p[1]),{"ships":0.0,"prod":0.0,"planets":0.0}); r["ships"]+=max(0.0,safe_float(p[5])); r["prod"]+=max(0.0,safe_float(p[6])); r["planets"]+=1.0
    return totals
def _fleet_owner(fleet:Any)->int|None:
    if isinstance(fleet,dict):
        for k in ("owner","player","player_id"):
            if k in fleet: return int(fleet[k])
        return None
    return int(fleet[1]) if isinstance(fleet,(list,tuple)) and len(fleet)>=2 else None
def _fleet_ships(fleet:Any)->float:
    if isinstance(fleet,dict):
        for k in ("ships","num_ships","ship_count"):
            if k in fleet: return max(0.0,safe_float(fleet[k]))
        return 0.0
    if _is_native_fleet_observation(fleet): return max(0.0,safe_float(fleet[5]))
    if isinstance(fleet,(list,tuple)):
        for i in (5,4,3):
            if len(fleet)>i and safe_float(fleet[i],-1.0)>=0.0: return safe_float(fleet[i])
    return 0.0
def _fleet_target_id(fleet:Any)->int|None:
    if isinstance(fleet,dict):
        for k in ("target_planet_id","target","to_planet_id","destination"):
            if k in fleet and fleet[k] is not None: return int(fleet[k])
        return None
    if _is_native_fleet_observation(fleet): return None
    if isinstance(fleet,(list,tuple)):
        for i in (3,2):
            if len(fleet)>i:
                try: return int(fleet[i])
                except Exception: pass
    return None
def _fleet_eta(fleet:Any)->float:
    if isinstance(fleet,dict):
        for k in ("eta","remaining_turns","turns_remaining","remaining"):
            if k in fleet: return safe_float(fleet[k],math.inf)
        return math.inf
    if _is_native_fleet_observation(fleet): return math.inf
    if isinstance(fleet,(list,tuple)):
        for i in (6,7,8):
            if len(fleet)>i and math.isfinite(safe_float(fleet[i],math.inf)): return safe_float(fleet[i],math.inf)
    return math.inf
def _native_fleet_movement(obs:dict[str,Any],player_id:int,*,horizon:int)->Any:
    if not any(_is_native_fleet_observation(f) for f in (obs.get("fleets",[]) or [])): return None
    try:
        g=make_geometry(horizon=int(horizon),device="cpu"); return g.build_or_update_movement(g.obs_to_tensors(obs,player_id=int(player_id)))
    except Exception: return None
def _fleet_target_and_eta(fleet:Any,movement:Any,*,horizon:int)->tuple[int|None,float]:
    tid=_fleet_target_id(fleet); eta=_fleet_eta(fleet)
    if (tid is None or not math.isfinite(eta)) and movement is not None:
        dt,de=_native_fleet_target_and_eta(fleet,movement,horizon=int(horizon)); tid=dt if tid is None else tid; eta=de if not math.isfinite(eta) else eta
    return tid,eta
def defaultdict_floats()->dict[str,float]: return {"enemy_5":0.0,"enemy_10":0.0,"enemy_20":0.0,"friendly_5":0.0,"friendly_10":0.0,"friendly_20":0.0}
def _incoming_by_slot(obs:dict[str,Any],player_id:int,max_planets:int)->dict[int,dict[str,float]]:
    id_to_slot={int(p[0]):i for i,p in enumerate(obs.get("planets",[])[:max_planets]) if len(p)>=7}; out={i:defaultdict_floats() for i in range(max_planets)}; movement=_native_fleet_movement(obs,int(player_id),horizon=20)
    for f in obs.get("fleets",[]) or []:
        tid,eta=_fleet_target_and_eta(f,movement,horizon=20)
        if tid not in id_to_slot or not math.isfinite(eta): continue
        slot=id_to_slot[tid]; prefix="friendly" if _fleet_owner(f)==int(player_id) else "enemy"; ships=_fleet_ships(f)
        for h in (5,10,20):
            if eta<=h: out[slot][f"{prefix}_{h}"]+=ships
    return out
def is_orbiting_planet(p:list[Any],initial_by_id:dict[int,list[Any]]|None=None)->bool:
    if len(p)<7: return False
    base=initial_by_id.get(int(p[0]),p) if initial_by_id else p; dx=safe_float(base[2])-CENTER; dy=safe_float(base[3])-CENTER; r=safe_float(base[4]); orbital=math.sqrt(dx*dx+dy*dy)
    return orbital+r<ROTATION_RADIUS_LIMIT and orbital>0.5
def planet_features(obs:dict[str,Any],player_id:int,slot:int,max_planets:int=P_MAX,incoming_by_slot:dict[int,dict[str,float]]|None=None)->list[float]:
    planets=obs.get("planets",[])[:max_planets]
    if slot<0 or slot>=len(planets) or len(planets[slot])<7: return [0.0]*len(PLANET_FEATURE_NAMES)
    p=planets[slot]; owner=int(p[1]); rel=relative_owner(owner,player_id); x=safe_float(p[2]); y=safe_float(p[3]); dx=x-CENTER; dy=y-CENTER; ships=max(0.0,safe_float(p[5])); prod=max(0.0,safe_float(p[6])); totals=_owner_totals(obs); ot=totals.get(owner,{"ships":0.0,"prod":0.0}) if owner>=0 else {"ships":0.0,"prod":0.0}
    inc=(incoming_by_slot if incoming_by_slot is not None else _incoming_by_slot(obs,player_id,max_planets)).get(slot,defaultdict_floats()); proj=ships+inc["friendly_20"]-inc["enemy_20"] if rel==1 else ships+inc["enemy_20"]-inc["friendly_20"]
    init={int(z[0]):z for z in obs.get("initial_planets",[]) if len(z)>=7}; comet=set(int(z) for z in obs.get("comet_planet_ids",[]) if int(z)>=0)
    row=[1.0,1.0 if rel==0 else 0.0,1.0 if rel==1 else 0.0,1.0 if rel==-1 else 0.0,dx/BOARD,dy/BOARD,safe_float(p[4])/5.0,math.log1p(ships)/SHIP_LOG_DENOM,prod/5.0,1.0 if int(p[0]) in comet else 0.0,1.0 if is_orbiting_planet(p,init) else 0.0,math.hypot(dx,dy)/(math.sqrt(2.0)*BOARD),ships/max(1.0,ot["ships"]),prod/max(1.0,ot["prod"]),proj/100.0,1.0 if inc["enemy_20"]>ships+inc["friendly_20"] else 0.0]
    return [float(x) for x in np.nan_to_num(np.asarray(row,dtype=np.float32),nan=0.0,posinf=0.0,neginf=0.0)]
def all_planet_features(obs:dict[str,Any],player_id:int,max_planets:int=P_MAX)->np.ndarray:
    inc=_incoming_by_slot(obs,player_id,max_planets); return np.asarray([planet_features(obs,player_id,i,max_planets,incoming_by_slot=inc) for i in range(max_planets)],dtype=np.float32)
def global_features(obs:dict[str,Any],player_id:int,max_planets:int=P_MAX)->np.ndarray:
    totals=_owner_totals(obs); players=_num_players(obs); my=totals.get(int(player_id),{"ships":0.0,"prod":0.0,"planets":0.0}); ts=sum(v["ships"] for v in totals.values()); tp=sum(v["prod"] for v in totals.values()); tpl=sum(v["planets"] for v in totals.values()); leader_s=max((v["ships"] for v in totals.values()),default=0.0); leader_p=max((v["prod"] for v in totals.values()),default=0.0); enemy=[v["ships"] for o,v in totals.items() if o!=int(player_id)]; weak=min(enemy,default=0.0); s=_step_norm(obs)
    arr=np.asarray([s,max(0.0,1.0-s),1.0 if players<=2 else 0.0,1.0 if players>=4 else 0.0,my["ships"]/max(1.0,ts),my["prod"]/max(1.0,tp),my["planets"]/max(1.0,tpl),(my["ships"]-leader_s)/max(1.0,ts),(my["prod"]-leader_p)/max(1.0,tp),(my["ships"]-weak)/max(1.0,ts)],dtype=np.float32)
    return np.nan_to_num(arr,nan=0.0,posinf=0.0,neginf=0.0)
def target_state_features(obs:dict[str,Any],player_id:int,max_planets:int=P_MAX)->np.ndarray:
    planets=obs.get("planets",[])[:max_planets]; incs=_incoming_by_slot(obs,player_id,max_planets); out=np.zeros((max_planets,len(TARGET_STATE_FEATURE_NAMES)),dtype=np.float32); own=[i for i,p in enumerate(planets) if len(p)>=7 and int(p[1])==int(player_id)]; enemy=[i for i,p in enumerate(planets) if len(p)>=7 and int(p[1])>=0 and int(p[1])!=int(player_id)]
    for tslot in range(max_planets):
        if tslot>=len(planets) or len(planets[tslot])<7: continue
        target=planets[tslot]; tx,ty=safe_float(target[2]),safe_float(target[3])
        def nearest(slots:list[int])->float:
            b=math.inf
            for s in slots:
                p=planets[s]; b=min(b,math.hypot(safe_float(p[2])-tx,safe_float(p[3])-ty)/10.0)
            return 0.0 if not math.isfinite(b) else min(1.0,b/50.0)
        oe,ee=nearest(own),nearest(enemy); inc=incs.get(tslot,defaultdict_floats()); owner=relative_owner(int(target[1]),player_id); proj=safe_float(target[5])+inc["friendly_20"]-inc["enemy_20"]
        out[tslot]=np.asarray([oe,ee,1.0 if ee<oe else 0.0,inc["enemy_10"]/100.0,float(owner if proj>0 else -owner),proj/100.0,1.0 if inc["friendly_10"]>0 and inc["enemy_10"]>0 else 0.0,1.0 if int(target[1])<0 and safe_float(target[5])<=5.0 else 0.0,1.0 if safe_float(target[6])>=3.0 else 0.0],dtype=np.float32)
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)
def build_feature_state(obs:dict[str,Any],player_id:int,max_planets:int=P_MAX)->FeatureState: return FeatureState(planet_features=all_planet_features(obs,player_id,max_planets),global_features=global_features(obs,player_id,max_planets),target_state_features=target_state_features(obs,player_id,max_planets))
def pair_features_from_dense(planet_features:np.ndarray,target_state_features:np.ndarray,source_slot:int,*,max_planets:int=P_MAX,target_viability_mask:np.ndarray|None=None,amount_viability_mask:np.ndarray|None=None)->np.ndarray:
    out=np.zeros((max_planets+1,len(PAIR_FEATURE_NAMES)),dtype=np.float32); ni={n:i for i,n in enumerate(PLANET_FEATURE_NAMES)}; ti={n:i for i,n in enumerate(TARGET_STATE_FEATURE_NAMES)}
    if not (0<=int(source_slot)<max_planets): out[NOOP_TARGET_SLOT,-1]=1.0; return out
    src=planet_features[int(source_slot)]; sx=float(src[ni["x_centered"]]); sy=float(src[ni["y_centered"]]); ss=max(0.0,ships_from_log_norm(float(src[ni["ships_log_norm"]]))); sp=max(0.0,float(src[ni["production_norm"]])*5.0); threat=float(src[ni["under_threat_20"]]); abc=max(1,int(np.asarray(amount_viability_mask).shape[-1])-1) if amount_viability_mask is not None else 0
    for tslot in range(max_planets):
        tgt=planet_features[tslot]
        if float(tgt[ni["alive"]])<=0.0: continue
        dx=float(tgt[ni["x_centered"]])-sx; dy=float(tgt[ni["y_centered"]])-sy; dist=math.hypot(dx,dy); ang=math.atan2(dy,dx) if dist>0 else 0.0; ts=max(0.0,ships_from_log_norm(float(tgt[ni["ships_log_norm"]]))); tp=max(0.0,float(tgt[ni["production_norm"]])*5.0); own=float(tgt[ni["rel_owner_own"]]); enemy=float(tgt[ni["rel_owner_enemy"]]); neutral=float(tgt[ni["rel_owner_neutral"]]); need=1.0 if own>0.5 else ts+1.0; safe=max(0.0,ss-(2.0+sp+10.0*threat)); rowts=target_state_features[tslot]; ne=float(rowts[ti["nearest_enemy_eta_to_target"]]); arrival=min(1.0,(dist*10.0)/50.0); proj=float(rowts[ti["projected_garrison_20"]]); gv=1.0 if target_viability_mask is not None and bool(np.asarray(target_viability_mask)[tslot]) else 0.0; gaf=float(np.asarray(amount_viability_mask)[tslot,1:].sum())/float(abc) if amount_viability_mask is not None and abc>0 else 0.0
        out[tslot]=np.asarray([need/100.0,need/max(1.0,ss),(ss-need)/100.0,tp/max(1.0,need),neutral,enemy,own,1.0 if neutral>0.5 and need<=5.0 else 0.0,1.0 if tp>=3.0 else 0.0,dist,math.sin(ang),math.cos(ang),gv,gaf,1.0 if gaf<=0.0 else 0.0,safe/100.0,(ss-need)/max(1.0,ss),ss/max(1.0,need),1.0 if ne<arrival else 0.0,ne-arrival,1.0 if ne<arrival and enemy>0.5 else 0.0,(ss/100.0)-proj,proj,float(rowts[ti["projected_owner_20"]]),0.0],dtype=np.float32)
    out[NOOP_TARGET_SLOT,-1]=1.0; return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)
