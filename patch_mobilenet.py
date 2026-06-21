import zipfile
import json
import os
import shutil

keras_file = 'fruit_8class_mobilenetv2_best.keras'
fixed_file = 'fruit_8class_mobilenetv2_best_fixed.keras'
temp_dir = 'temp_mobilenet_extract'

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

print("Extracting model...")
with zipfile.ZipFile(keras_file, 'r') as z:
    z.extractall(temp_dir)

config_path = os.path.join(temp_dir, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

layers = cfg['config']['layers']

def clean_inbound_nodes(inbound_nodes):
    cleaned = []
    for node in inbound_nodes:
        if isinstance(node, dict):
            kwargs = node.get('kwargs', {})
            for key in ['mask', 'training']:
                if key in kwargs:
                    kwargs.pop(key)
            node['kwargs'] = kwargs
        cleaned.append(node)
    return cleaned

def patch_layer(layer):
    name = layer.get('name', '')
    cls = layer.get('class_name', '')

    if cls == 'Lambda' and name == 'to_float32':
        print(f"  [FIX] Replacing Lambda '{name}' with Activation linear")
        original_inbound = clean_inbound_nodes(layer.get('inbound_nodes', []))
        input_shape = layer.get('build_config', {}).get('input_shape', [None, 224, 224, 3])
        return {
            "module": "keras.layers",
            "class_name": "Activation",
            "config": {
                "name": name,
                "trainable": True,
                "dtype": "float32",
                "activation": "linear"
            },
            "registered_name": None,
            "build_config": {"input_shape": input_shape},
            "name": name,
            "inbound_nodes": original_inbound
        }

    layer_cfg = layer.get('config', {})
    if 'quantization_config' in layer_cfg:
        print(f"  [FIX] Removing 'quantization_config' from '{name}'")
        del layer_cfg['quantization_config']

    if isinstance(layer_cfg.get('dtype'), dict):
        layer_cfg['dtype'] = layer_cfg['dtype'].get('config', {}).get('name', 'float32')

    if 'inbound_nodes' in layer:
        layer['inbound_nodes'] = clean_inbound_nodes(layer['inbound_nodes'])

    return layer

print("Patching layers...")
count = 0
for i, layer in enumerate(layers):
    layers[i] = patch_layer(layer)
    count += 1
    if isinstance(layers[i].get('config'), dict) and 'layers' in layers[i]['config']:
        sub_layers = layers[i]['config']['layers']
        for j, sub_layer in enumerate(sub_layers):
            sub_layers[j] = patch_layer(sub_layer)
            count += 1

print(f"Patched {count} layers total.")

print("Writing new config.json...")
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"Repacking into {fixed_file}...")
if os.path.exists(fixed_file):
    os.remove(fixed_file)

with zipfile.ZipFile(fixed_file, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir).replace('\\', '/')
            z.write(file_path, arcname)

shutil.rmtree(temp_dir)
print(f"\nDone! Fixed file: {fixed_file}")
