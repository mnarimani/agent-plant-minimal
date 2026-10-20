"""Original XML systems for presentation, independent of flattened execution edges."""
from __future__ import annotations
import json
import re
from xml.etree import ElementTree as ET
from .viewer_source import _numbers, _parameter
from .source_geometry import native_layout

VISUAL_TYPES = {'Inport', 'Outport', 'From', 'Goto', 'GotoTagVisibility', 'SubSystem',
                'Gain', 'Constant', 'Sum', 'Product', 'Integrator', 'TransferFcn',
                'Saturate', 'Step', 'DiscretePulseGenerator', 'ManualSwitch', 'Mux',
                'Demux', 'Fcn', 'Lookup_n-D', 'Scope', 'Terminator', 'EnablePort', 'TriggerPort'}
ALIASES = {'SubSystem':'Subsystem', 'DiscretePulseGenerator':'PulseGenerator', 'Lookup_n-D':'Lookup1D'}


def xml_views(topology, runtime_graph):
    if topology is None or topology.source_directory is None:
        raise ValueError('Original XML is required for a source diagram. Convert the SLX again.')
    directory = topology.source_directory/'simulink/systems'
    runtime = {node['id']:node for node in runtime_graph['nodes']}
    views, visiting = {}, set()

    def read_system(filename, path=(), names=()):
        key = json.dumps(path) if path else 'overview'
        if filename in visiting:
            raise ValueError('Recursive XML system references cannot be drawn.')
        visiting.add(filename)
        xml = ET.parse(directory/filename).getroot()
        nodes, by_sid = [], {}
        for block in xml.findall('Block'):
            sid, name, original_type = block.get('SID'), block.get('Name',''), block.get('BlockType')
            if not sid or sid in by_sid or original_type not in VISUAL_TYPES:
                raise ValueError(f'No source diagram definition for {original_type!r} ({name!r}), or invalid SID.')
            identifier = '::'.join((*path,sid))
            sf = _parameter(block,'SFBlockType')
            kind = 'StateflowChart' if sf=='Chart' else 'MatlabFunction' if sf=='MATLAB Function' else ALIASES.get(original_type,original_type)
            params = dict(runtime.get(identifier,{}).get('parameters',{}))
            params['source_parameters'] = {item.get('Name'):item.text for item in block.findall('P')
                                           if item.get('Name') not in ('Position','ZOrder','BlockRotation','BlockMirror') and item.text}
            if kind in ('From','Goto','GotoTagVisibility'): params['tag'] = _parameter(block,'GotoTag','A')
            if kind=='Goto': params['visibility'] = _parameter(block,'TagVisibility','local')
            if kind=='Sum' and 'signs' not in params:
                params['signs'] = [1 if sign=='+' else -1 for sign in _parameter(block,'Inputs','++') if sign in '+-']
            counts = block.find('PortCounts')
            bounds = _numbers(_parameter(block,'Position'))
            if len(bounds)!=4 or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
                raise ValueError(f'Missing or invalid XML Position for {name!r}; no guessed placement was generated.')
            node = {'id':identifier,'name':'/'.join((*names,name)), 'display_name':name,'type':kind,
                    'parameters':params,'original_type':original_type,'source_number':_parameter(block,'Port','1'),
                    'source_layout':{'bounds':bounds,'rotation':int(_parameter(block,'BlockRotation','0'))%360,
                                     'mirror':_parameter(block,'BlockMirror','off')=='on',
                                     'name_location':_parameter(block,'NameLocation','bottom'),
                                     'show_name':_parameter(block,'ShowName','on')!='off',
                                     'icon_shape':_parameter(block,'IconShape','round'),
                                     'sum_inputs':_parameter(block,'Inputs','++'),
                                     'input_count':int(counts.get('in','0')) if counts is not None else 0,
                                     'output_count':int(counts.get('out','0')) if counts is not None else 0}}
            if kind=='Subsystem':
                system = block.find('System')
                ref = system.get('Ref') if system is not None else None
                if not ref or not re.fullmatch(r'system_[\w-]+',ref):
                    raise ValueError(f'Missing or invalid XML system reference for {name!r}.')
                node['child_view'] = read_system(ref+'.xml',(*path,sid),(*names,name))
            nodes.append(node); by_sid[sid]=node
        edges = []
        def endpoint(text, direction):
            match = re.fullmatch(r'(.+)#(in|out):(\d+)',text)
            if match and match[2]==direction and int(match[3])>0 and match[1] in by_sid:
                return {'block_id':by_sid[match[1]]['id'],'port':int(match[3])-1}
            named = re.fullmatch(r'(.+)#(enable|trigger|reset|ifaction)',text)
            if named and direction=='in' and named[1] in by_sid:
                return {'block_id':by_sid[named[1]]['id'],'port':named[2]}
            raise ValueError(f'Unknown XML signal endpoint {text!r}; no connection was guessed.')
        def branches(element, source, offsets=(), label=''):
            raw = _numbers(_parameter(element,'Points'))
            if len(raw)%2: raise ValueError('Invalid XML line Points.')
            points = [*offsets,*[raw[i:i+2] for i in range(0,len(raw),2)]]
            target = _parameter(element,'Dst')
            if target:
                destination = endpoint(target,'in')
                edges.append({'source':source,'destination':destination,
                              'kind':'signal' if isinstance(destination['port'],int) else destination['port'],
                              'offsets':points,'label':label})
            for branch in element.findall('Branch'): branches(branch,source,points,label)
        for line in xml.findall('Line'):
            branches(line,endpoint(_parameter(line,'Src'),'out'),label=_parameter(line,'Name'))
        view = {'title':' / '.join(names) or 'Model overview','nodes':nodes,'edges':edges,'source_system':filename}
        view['layouts'] = {'native':native_layout(view)}
        views[key]=view
        visiting.remove(filename)
        return key
    read_system('system_root.xml')
    return dict(sorted(views.items(),key=lambda pair:(pair[0]!='overview',pair[1]['title'])))
