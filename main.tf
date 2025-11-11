terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.5.0"
}

provider "aws" {
  region = "us-east-2"
}

resource "aws_vpc" "dl_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "dl-vpc"
  }
}

resource "aws_subnet" "dl_subnet" {
  vpc_id                  = aws_vpc.dl_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "us-east-2a"
  tags = {
    Name = "dl-subnet"
  }
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Permitir acceso SSH"
  vpc_id      = aws_vpc.dl_vpc.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# Instancia EC2 Deep Learning ,   ami                    = "ami-073e1d05d567eab85" , instance_type = "g5.xlarge", g4dn.xlarge

resource "aws_instance" "dl_training" {
  ami                    = "ami-073e1d05d567eab85"
  instance_type          = "g5.xlarge"
  key_name               = "aws_ssh_key"  
  subnet_id              = aws_subnet.dl_subnet.id
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

    user_data = <<-EOF
    #!/bin/bash
    # Ejecutar todo como usuario ubuntu
    sudo -i -u ubuntu bash << 'EOC'

    # Actualizar sistema
    sudo apt update -y
    sudo apt upgrade -y
    sudo apt install -y git htop nvtop python3-venv unzip curl

    # Instalar AWS CLI v2
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws

    # Clonar repo solo si no existe
    cd /home/ubuntu
    if [ ! -d "DL_algorithm_bean" ]; then
        git clone https://github.com/Anthonymss/DL_algorithm_bean.git
    fi
    cd DL_algorithm_bean

    # Crear entorno virtual Python
    python3 -m venv venv
    source venv/bin/activate

    # Instalar requirements si existe el archivo
    if [ -f requirements.txt ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    fi

    # Marcar instancia lista
    echo "Deep Learning instance ready!  source venv/bin/activate" > /home/ubuntu/READY.txt

    EOC
    EOF


  tags = {
    Name    = "DL-Training"
    Project = "DeepLearning"
  }
}


output "instance_public_ip" {
  description = "IP pública de la instancia"
  value       = aws_instance.dl_training.public_ip
}

output "ssh_connection_command" {
  description = "Comando SSH para conectarte"
  value       = "ssh -i ~/.ssh/aws_ssh_key.pem ubuntu@${aws_instance.dl_training.public_ip}"
}
