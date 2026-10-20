"""Keep XML block rectangles and line offsets; never run a graph-layout algorithm."""
from math import cos, sin, pi
from .diagram_layout import SIDES, _anchor

SCALE = 2.65


def native_layout(view):
    if not view['nodes']:
        return {'nodes':{},'edges':[],'bounds':[0,0,400,200],'origin':[0,0],'scale':SCALE,'source':'xml','empty':True}
    counts = {node['id']:[node['source_layout']['input_count'],node['source_layout']['output_count']] for node in view['nodes']}
    for edge in view['edges']:
        for direction,key in ((0,'destination'),(1,'source')):
            endpoint=edge[key]
            if isinstance(endpoint['port'],int): counts[endpoint['block_id']][direction]=max(counts[endpoint['block_id']][direction],endpoint['port']+1)
    boxes = [node['source_layout']['bounds'] for node in view['nodes']]
    origin = [min(b[0] for b in boxes),min(b[1] for b in boxes)]
    geometries={}
    for node in view['nodes']:
        h=node['source_layout']; l,t,r,b=h['bounds']; ins,outs=counts[node['id']]
        sides={'input':['left']*ins,'output':['right']*outs}
        if node['type']=='Sum' and len(h['sum_inputs'])==3 and '|' in h['sum_inputs']:
            sides['input']=[side for side,sign in zip(('top','left','bottom'),h['sum_inputs']) if sign!='|']
        if node['type'] in ('EnablePort','TriggerPort'): sides['output']=['bottom']*outs
        for direction in sides:
            sides[direction]=[SIDES[(SIDES.index(side)+h['rotation']//90)%4] for side in sides[direction]]
            if h['mirror']: sides[direction]=[{'left':'right','right':'left'}.get(side,side) for side in sides[direction]]
        g={'x':(l-origin[0])*SCALE+70,'y':(t-origin[1])*SCALE+70,'width':(r-l)*SCALE,'height':(b-t)*SCALE,
           'label':node['display_name'] if h['show_name'] else '', 'label_width':max((r-l)*SCALE,len(node['display_name'])*7),
           'label_side':h['name_location'],'sides':sides,'rotation':h['rotation']}
        g['ports']={direction:[_anchor(g,index,direction) for index in range(len(values))] for direction,values in sides.items()}
        # Multi-input circular sums place ports around the circle, not evenly on a rectangle.
        if node['type']=='Sum' and h['icon_shape']!='rectangular' and h['sum_inputs'].startswith('|') and len(h['sum_inputs'])>3:
            angles = [pi*.75+pi*.5*i/max(1,ins-1) for i in range(ins)]
            for port,angle in zip(g['ports']['input'],angles):
                x,y=cos(angle)*g['width']/2,sin(angle)*g['height']/2
                # Canvas y increases downwards; the first sign is the upper-left input.
                y=-y
                for _ in range(h['rotation']//90): x,y=-y,x
                if h['mirror']: x=-x
                port.update(x=g['x']+g['width']/2+x,y=g['y']+g['height']/2+y)
        geometries[node['id']]=g
    routes=[]
    for index,edge in enumerate(view['edges']):
        source=geometries[edge['source']['block_id']]; destination=geometries[edge['destination']['block_id']]
        a=source['ports']['output'][edge['source']['port']]
        port=edge['destination']['port']
        end=destination['ports']['input'][port] if isinstance(port,int) else {'x':destination['x']+destination['width']/2,'y':destination['y'],'side':'top'}
        points=[[a['x'],a['y']]]
        for dx,dy in edge['offsets']: points.append([points[-1][0]+dx*SCALE,points[-1][1]+dy*SCALE])
        x,y=points[-1]; target=[end['x'],end['y']]
        if abs(x-target[0])>1e-8 and abs(y-target[1])>1e-8:
            # Only complete the last approach. All saved XML segments stay intact.
            points.append([x,target[1]] if end['side'] in ('left','right') else [target[0],y])
        points.append(target)
        routes.append({'index':index,'points':points,'label':edge['label'],'fallback':False,'origin':'xml'})
    xs=[g['x'] for g in geometries.values()]+[g['x']+g['width'] for g in geometries.values()]
    ys=[g['y']-25 for g in geometries.values()]+[g['y']+g['height']+30 for g in geometries.values()]
    for route in routes:
        xs.extend(p[0] for p in route['points']);ys.extend(p[1] for p in route['points'])
    return {'nodes':geometries,'edges':routes,'bounds':[min(xs)-45,min(ys)-45,max(xs)-min(xs)+90,max(ys)-min(ys)+90],
            'origin':origin,'scale':SCALE,'source':'xml'}
