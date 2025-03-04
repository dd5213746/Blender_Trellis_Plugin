bl_info = {
    'name': 'Trellis_3D',  # 插件名称
    'blender': (4, 3, 2),  # 插件所兼容的版本
    'author': 'winterding',  # 作者名
    'version': (0, 4, 0),  # 插件版本号
    'description': "Trellis AI建模Blender插件"  # 插件功能说明
}

import bpy
import sys
import os
import asyncio
import time
import threading
import subprocess
from gradio_client import Client, handle_file


async def trellis_generate(client_url, image_path):
    client = Client(client_url)
    temp_image = client.predict(
        image=handle_file(image_path),
        api_name="/preprocess_image_1"
    )
    client.predict(
        api_name="/start_session"
    )

    result = client.predict(
        randomize_seed=True,
        seed=0,
        api_name="/get_seed"
    )

    client.predict(
        image=handle_file(temp_image),
        multiimages=[],
        seed=result,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3,
        slat_sampling_steps=12,
        multiimage_algo="stochastic",
        api_name="/image_to_3d"
    )

    result = client.predict(
        mesh_simplify=0.95,
        texture_size=1024,
        api_name="/extract_glb"
    )
    return result


async def trellis_multi_generate(client_url, image_path1, image_path2, image_path3):
    client = Client(client_url)

    client.predict(
        api_name="/start_session"
    )

    image_list = [{"image": handle_file(image_path1), "caption": None},
                  {"image": handle_file(image_path2), "caption": None},
                  {"image": handle_file(image_path3), "caption": None}, ]

    temp_images = client.predict(
        images=image_list,
        api_name="/preprocess_images"
    )

    result = client.predict(
        randomize_seed=True,
        seed=0,
        api_name="/get_seed"
    )

    client.predict(
        image=handle_file(image_path1),
        multiimages=image_list,
        seed=result,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3,
        slat_sampling_steps=12,
        multiimage_algo="multidiffusion",
        api_name="/image_to_3d"
    )

    client.predict(
        api_name="/start_session"
    )

    result = client.predict(
        mesh_simplify=0.95,
        texture_size=1024,
        api_name="/extract_glb"
    )
    return result


class ComfirmLocalHostOperator(bpy.types.Operator):
    bl_idname = "wm.confirm_localhost"
    bl_label = "Custom Button Operator"

    async def async_execute(self, context):
        client_url = context.scene.localhost
        if (context.scene.multi_images):
            image_path_1 = context.scene.preview_image_1.filepath
            image_path_2 = context.scene.preview_image_2.filepath
            image_path_3 = context.scene.preview_image_3.filepath
            result = await trellis_multi_generate(client_url, image_path_1, image_path_2, image_path_3)
        else:
            image_path = context.scene.preview_image.filepath
            result = await trellis_generate(client_url, image_path)
        if (result):
            gltf_path = str(result[0])
            bpy.ops.import_scene.gltf(filepath=gltf_path)
            context.scene.generate_message = "没有进行中的生成任务"
        return {'FINISHED'}

    def execute(self, context):
        context.scene.generate_message = "模型生成中，请稍后..."
        thread = threading.Thread(target=lambda: asyncio.run(self.async_execute(context)))
        thread.start()
        return {'FINISHED'}


class SwitchGenerateMode(bpy.types.Operator):
    bl_idname = "wm.switch_mode"
    bl_label = "SwitchGenerateMode"

    def execute(self, context):
        context.scene.multi_images = not context.scene.multi_images
        return {'FINISHED'}


class ModelFixOperator(bpy.types.Operator):
    bl_idname = "wm.model_fix"
    bl_label = "Model Fix Button"

    def execute(self, context):
        selected_objects = bpy.context.selected_objects

        for obj in selected_objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=0.005)
                bpy.ops.mesh.tris_convert_to_quads(uvs=True)
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.shade_auto_smooth()
                for modifier in obj.modifiers:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
        return {'FINISHED'}


class LoadImageOperator(bpy.types.Operator):
    bl_idname = "wm.load_image"
    bl_label = "Load Image from Path"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        scn = context.scene
        path = self.filepath

        # 检查路径是否有效
        if not os.path.exists(path):
            self.report({'ERROR'}, "Invalid image path")
            return {'CANCELLED'}

        # 加载图像
        try:
            image = bpy.data.images.load(path)
            scn.preview_image = image
            scn.image_path = path  # 更新路径
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class LoadRightImageOperator(bpy.types.Operator):
    bl_idname = "wm.load_image_1"
    bl_label = "Load Image from Path"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        scn = context.scene
        path = self.filepath

        # 检查路径是否有效
        if not os.path.exists(path):
            self.report({'ERROR'}, "Invalid image path")
            return {'CANCELLED'}

        # 加载图像
        try:
            image = bpy.data.images.load(path)
            scn.preview_image_1 = image
            scn.image_path_1 = path  # 更新路径
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class LoadFrontImageOperator(bpy.types.Operator):
    bl_idname = "wm.load_image_2"
    bl_label = "Load Image from Path"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        scn = context.scene
        path = self.filepath

        # 检查路径是否有效
        if not os.path.exists(path):
            self.report({'ERROR'}, "Invalid image path")
            return {'CANCELLED'}

        # 加载图像
        try:
            image = bpy.data.images.load(path)
            scn.preview_image_2 = image
            scn.image_path_2 = path  # 更新路径
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class LoadBackImageOperator(bpy.types.Operator):
    bl_idname = "wm.load_image_3"
    bl_label = "Load Image from Path"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')

    def execute(self, context):
        scn = context.scene
        path = self.filepath

        # 检查路径是否有效
        if not os.path.exists(path):
            self.report({'ERROR'}, "Invalid image path")
            return {'CANCELLED'}

        # 加载图像
        try:
            image = bpy.data.images.load(path)
            scn.preview_image_3 = image
            scn.image_path_3 = path  # 更新路径
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def load_venv():
    python_exe = os.path.join(
        sys.prefix, 'bin', 'python.exe'
    )
    subprocess.call(
        [python_exe, "-m", "ensurepip"]
    )
    subprocess.call(
        [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
    )
    subprocess.call(
        [python_exe, "-m", "pip", "install", "gradio_client"]
    )
    subprocess.call(
        [python_exe, "-m", "pip", "install", "file"]
    )


class LoadVenvOperator(bpy.types.Operator):
    bl_idname = "wm.load_venv"
    bl_label = "Load Venv"

    def execute(self, context):
        load_venv()
        return {'FINISHED'}


class TrellisUIPanel(bpy.types.Panel):
    bl_idname = "Trellis_Manage_panel"
    bl_label = "Trellis Plugin"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Trellis Model Generator"

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene

        row = layout.row()
        row.label(text="LocalHost:")
        row.prop(context.scene, "localhost", text="")
        row = layout.row()
        row.operator("wm.load_venv", text="配置运行环境")

        row = layout.row()
        row = layout.row()
        row = layout.row()
        row = layout.row()

        box = layout.box()
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="TextureToModel")
        row = box.row()
        row.operator("wm.switch_mode", text="switch")
        row = box.row()

        if (context.scene.multi_images):
            row.template_ID_preview(context.scene, "preview_image_1", open="wm.load_image_1")
            row.template_ID_preview(context.scene, "preview_image_2", open="wm.load_image_2")
            row.template_ID_preview(context.scene, "preview_image_3", open="wm.load_image_3")
        else:
            row.template_ID_preview(context.scene, "preview_image", open="wm.load_image")

        if (context.scene.multi_images):
            if scene.preview_image_1 and scene.preview_image_2 and scene.preview_image_3:
                row = box.row()
                row = box.row()
                row.operator("wm.confirm_localhost", text="Generate")
                row = box.row()
                box.label(text=context.scene.generate_message)
        else:
            if scene.preview_image:
                row = box.row()
                row = box.row()
                row.operator("wm.confirm_localhost", text="Generate")
                row = box.row()
                box.label(text=context.scene.generate_message)

        row = layout.row()
        row = layout.row()
        row = layout.row()
        row = layout.row()

        box = layout.box()
        row = box.row()
        row.alignment = 'CENTER'
        row.label(text="TrellisModelFixer")
        row = box.row()
        row.operator("wm.model_fix", text="优化模型")


bpy.types.Scene.generate_message = bpy.props.StringProperty(name="生成信息", default="没有进行中的生成任务")

bpy.types.Scene.localhost = bpy.props.StringProperty(name="文本输入", default="http://127.0.0.1:7860/")

bpy.types.Scene.preview_image = bpy.props.PointerProperty(name="单贴图输入", type=bpy.types.Image)
bpy.types.Scene.preview_image_1 = bpy.props.PointerProperty(name="多贴图输入_1", type=bpy.types.Image)
bpy.types.Scene.preview_image_2 = bpy.props.PointerProperty(name="多贴图输入_2", type=bpy.types.Image)
bpy.types.Scene.preview_image_3 = bpy.props.PointerProperty(name="多贴图输入_3", type=bpy.types.Image)

bpy.types.Scene.multi_images = bpy.props.BoolProperty(name="multi_image_info")

bpy.types.Scene.image_path = bpy.props.StringProperty(name="image_path", default="----")
bpy.types.Scene.image_path_1 = bpy.props.StringProperty(name="image_path_1", default="----")
bpy.types.Scene.image_path_2 = bpy.props.StringProperty(name="image_path_2", default="----")
bpy.types.Scene.image_path_3 = bpy.props.StringProperty(name="image_path_3", default="----")


def register():
    bpy.utils.register_class(ModelFixOperator)
    bpy.utils.register_class(LoadImageOperator)
    bpy.utils.register_class(LoadRightImageOperator)
    bpy.utils.register_class(LoadFrontImageOperator)
    bpy.utils.register_class(LoadBackImageOperator)
    bpy.utils.register_class(ComfirmLocalHostOperator)
    bpy.utils.register_class(TrellisUIPanel)
    bpy.utils.register_class(LoadVenvOperator)
    bpy.utils.register_class(SwitchGenerateMode)


def unregister():
    bpy.utils.unregister_class(ModelFixOperator)
    bpy.utils.unregister_class(LoadImageOperator)
    bpy.utils.unregister_class(LoadRightImageOperator)
    bpy.utils.unregister_class(LoadFrontImageOperator)
    bpy.utils.unregister_class(LoadBackImageOperator)
    bpy.utils.unregister_class(ComfirmLocalHostOperator)
    bpy.utils.unregister_class(TrellisUIPanel)
    bpy.utils.unregister_class(LoadVenvOperator)
    bpy.utils.unregister_class(SwitchGenerateMode)


if __name__ == "__main__":
    bpy.types.Scene.generate_message = bpy.props.StringProperty(name="生成信息", default="没有进行中的生成任务")
    register()