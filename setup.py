from setuptools import setup, find_packages

setup(name='diaryautocoder',
	  version='0.3',
	  author='Michell Li',
	  packages=find_packages(),
	  install_requires=['scikit-learn==0.19.2',
						'numpy',
						'pandas',
						'dash',
						'scipy'
						],
	  entry_points={
	  			'console_scripts': [
									'launch_diary_autocoder = ce_diary_autocoder.upload_component:main'
				]
	  },
	  include_package_data=True,
	  package_data={
	  			'': ['*.json', '*.xlsx', '*.txt'],
	  })
