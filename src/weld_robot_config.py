from dataclasses import dataclass
from pathlib import Path
import subprocess

import yaml


@dataclass(frozen=True)
class WeldRobotConfig:
    use_prismatic_joint: bool
    ee_link: str
    arm_accel_indices: tuple[int, ...]

    def generator_cmd(self, generator: Path) -> list[str]:
        cmd = ["python3", str(generator)]
        if self.use_prismatic_joint:
            cmd.append("--use-prismatic-joint")
        return cmd

    def robot_description(self, generator: Path) -> tuple[str, str]:
        cmd = self.generator_cmd(generator)
        urdf = subprocess.check_output([*cmd, "-o", "urdf"], text=True)
        srdf = subprocess.check_output([*cmd, "-o", "srdf"], text=True)
        return urdf, srdf

    def write_task_yaml(self, template: Path, output: Path) -> Path:
        with template.open() as file:
            task_yaml = yaml.safe_load(file)

        task_yaml["ee_pos"]["distal_link"] = self.ee_link
        task_yaml["ee_ori"]["distal_link"] = self.ee_link
        task_yaml["acceleration_regularization_arm"]["indices"] = list(
            self.arm_accel_indices)
        if not self.use_prismatic_joint:
            task_yaml["costs"] = [
                cost for cost in task_yaml["costs"]
                if cost not in (
                    "acceleration_regularization_prismatic_yaw",
                    "acceleration_regularization_prismatic_z",
                )
            ]
        with output.open("w") as file:
            yaml.safe_dump(task_yaml, file, sort_keys=False)
        return output


def weld_robot_config(use_prismatic_joint: bool) -> WeldRobotConfig:
    if use_prismatic_joint:
        return WeldRobotConfig(
            use_prismatic_joint=True,
            ee_link="ee_F",
            arm_accel_indices=(16, 17, 18, 19, 20, 21),
        )
    return WeldRobotConfig(
        use_prismatic_joint=False,
        ee_link="ee_E",
        arm_accel_indices=(14, 15, 16, 17, 18, 19, 20),
    )
