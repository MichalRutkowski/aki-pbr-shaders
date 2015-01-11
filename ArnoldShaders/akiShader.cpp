#include <ai.h>
#include <cstring>
#include <map>

#define _CRT_SECURE_NO_WARNINGS

// A Simple Shader
AI_SHADER_NODE_EXPORT_METHODS(SimpleMethods);

struct ShaderData
{
	AtSampler* diffuse_sampler;
	AtSampler* glossy_sampler;
	AtSampler* glossy2_sampler;
	AtSampler* refraction_sampler;
	AtSampler* backlight_sampler;
	int GI_diffuse_depth;
	int GI_reflection_depth;
	int GI_refraction_depth;
	int GI_glossy_depth;
	int GI_diffuse_samples;
	int GI_glossy_samples;
	int glossy_samples2;
	int diffuse_sample_offset;
	int glossy_sample_offset;
	int total_samples;
	AtCritSec cs;
	std::map<AtNode*, int> lightGroups;
	bool specular1NormalConnected;
	bool specular2NormalConnected;
	bool lightGroupsIndirect;
	bool standardAovs;
	int numLights;

	// AOV names
	std::string aov_diffuse_color;
	std::string aov_direct_diffuse;
	std::string aov_direct_diffuse_raw;
	std::string aov_indirect_diffuse;
	std::string aov_indirect_diffuse_raw;
	std::string aov_direct_specular;
	std::string aov_indirect_specular;
};

namespace
{

	enum SimpleParams { 
		in_color, 
		in_roughness,
		p_aov_diffuse_color,
		p_aov_direct_diffuse,
		p_aov_direct_diffuse_raw,
		p_aov_indirect_diffuse,
		p_aov_indirect_diffuse_raw,
		p_aov_direct_specular,
		p_aov_indirect_specular,
	};

};

node_parameters
{
	AiParameterRGB("baseColor", 0.5f, 0.5f, 0.5f);
	AiParameterFlt("roughness", 0.5f);
	AiParameterStr("aov_diffuse_color", "diffuse_color");
	AiParameterStr("aov_direct_diffuse", "direct_diffuse");
	AiParameterStr("aov_direct_diffuse_raw", "direct_diffuse_raw");
	AiParameterStr("aov_indirect_diffuse", "indirect_diffuse");
	AiParameterStr("aov_indirect_diffuse_raw", "indirect_diffuse_raw");
	AiParameterStr("aov_direct_specular", "direct_specular");
	AiParameterStr("aov_indirect_specular", "indirect_specular");


}

node_initialize
{
	ShaderData* data = new ShaderData;
	AiNodeSetLocalData(node, data);
	data->glossy_sampler = NULL;
}

node_update
{
	ShaderData *data = (ShaderData*)AiNodeGetLocalData(node);
	data->aov_diffuse_color = params[p_aov_diffuse_color].STR;
	data->aov_direct_diffuse = params[p_aov_direct_diffuse].STR;
	data->aov_direct_diffuse_raw = params[p_aov_direct_diffuse_raw].STR;
	data->aov_indirect_diffuse = params[p_aov_indirect_diffuse].STR;
	data->aov_indirect_diffuse_raw = params[p_aov_indirect_diffuse_raw].STR;
	data->aov_direct_specular = params[p_aov_direct_specular].STR;
	data->aov_indirect_specular = params[p_aov_indirect_specular].STR;


	AiAOVRegister(data->aov_diffuse_color.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_diffuse.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_diffuse_raw.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_diffuse.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_diffuse_raw.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_specular.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_specular.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
}

node_finish
{
	if (AiNodeGetLocalData(node))
	{
		ShaderData* data = (ShaderData*)AiNodeGetLocalData(node);
		AiSamplerDestroy(data->glossy_sampler);
		AiNodeSetLocalData(node, NULL);
		delete data;
	}
}

// based on Brian Karis' SIGGRAPH 2013 talk
void importanceSampleGGX(const AtVector sample, float roughness, AtVector N, AtVector &outH)
{
	float alpha2 = roughness * roughness;

	float phi = 2.f * AI_PI * sample.x;
	float cosTheta = sqrtf((1.f - sample.y) / (1.f + (alpha2 * alpha2 - 1.f) * sample.y));
	float sinTheta = sqrtf(1.f - cosTheta * cosTheta);

	AtVector H = AiVector((sinTheta * cosf(phi)), (sinTheta * sinf(phi)), cosTheta);

	AtVector upVector = abs(N.z) < 0.999f ? AiVector(0.f, 0.f, 1.f) : AiVector(1.f, 0.f, 0.f);
	AtVector tangentX = AiV3Normalize(AiV3Cross(upVector, N));
	AtVector tangentY = AiV3Cross(N, tangentX);

	float Hx, Hy, Hz;
	Hx = H.x;
	Hy = H.y;
	Hz = H.z;
	outH = (tangentX * AiVector(Hx, Hx, Hx)) + (tangentY * AiVector(Hy, Hy, Hy)) + (N * AiVector(Hz, Hz, Hz));
}

void computeSpecular(const float s_roughness, AtColor &s_direct_result, AtColor &s_indirect_result, AtShaderGlobals *sg)
{
	AiLightsPrepare(sg);
	AtVector x;
	// according to sdk docs, inputs 2,3 and 5 (anisotropy) are unused
	// however it seems they've already implemented two-dimensional roughness so docs are outdated
	void*			d_brdf_data = AiCookTorranceMISCreateData(sg, &x, &x, s_roughness, s_roughness);

	while (AiLightsGetSample(sg))
	{
		s_direct_result += AiEvaluateLightSample(sg, d_brdf_data, AiCookTorranceMISSample, AiCookTorranceMISBRDF, AiCookTorranceMISPDF) * AiLightGetSpecular(sg->Lp);
	}
	s_indirect_result = AiCookTorranceIntegrate(&sg->N, sg, &x, &x, s_roughness, s_roughness);
}

// based on kettle shaders
void Kettle_diffuse(const float d_roughness, AtColor &d_direct_result, AtColor &d_indirect_result, AtShaderGlobals *sg)
{
	// -------------
	// do light loop
	// -------------

	AiLightsPrepare(sg);
	void*			d_brdf_data = AiOrenNayarMISCreateData(sg, d_roughness);

	while (AiLightsGetSample(sg))
	{
		d_direct_result += AiEvaluateLightSample(sg, d_brdf_data, AiOrenNayarMISSample, AiOrenNayarMISBRDF, AiOrenNayarMISPDF) * AiLightGetDiffuse(sg->Lp);
	}

	// indirect diffuse

	d_indirect_result = AiOrenNayarIntegrate(&sg->Nf, sg, d_roughness);
};


shader_evaluate
{
	// read parametres
	AtColor Kd = AiShaderEvalParamRGB(in_color);
	float roughness = AiShaderEvalParamFlt(in_roughness);
	roughness = roughness * roughness;

	//clamp roughness to (0, 1)
	//roughness = roughness > 1.f ? 1.f : (roughness < 0.f ? 0.f : roughness);

	// view vector
	AtVector V = AiV3Normalize(-sg->Rd);
	AtVector N = sg->N;

	// DIFFUSE //
	AtColor diffuse_direct = AI_RGB_BLACK; 
	AtColor diffuse_indirect = AI_RGB_BLACK;
	Kettle_diffuse(roughness, diffuse_direct, diffuse_indirect, sg);

	// DIRECT SPECULAR REFLECTION
	AtColor specular_direct = AI_RGB_BLACK;
	AtColor specular_indirect = AI_RGB_BLACK;
	computeSpecular(roughness, specular_direct, specular_indirect, sg);

	// FRESNEL - at the moment it's N x V
	float fresnel = AiFresnelWeight(sg->N, sg->Rd, 0.04f);
	// let's try computing Fresnel using my custom GGX distribution


	// compute final result
	AtColor o_diffuse = Kd * (diffuse_direct + diffuse_indirect);	
	AtColor o_specular = fresnel * (specular_direct + specular_indirect);

	AtColor col = AiColorCreate(V.x, V.y, V.z);
	// no energy conservation at the moment?

	// WRITE AOVs
	ShaderData *data = (ShaderData*)AiNodeGetLocalData(node);
	if (Kd != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_diffuse_color.c_str(), Kd);
	if (diffuse_direct != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_direct_diffuse.c_str(), Kd * diffuse_direct);
	if (specular_direct != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_direct_specular.c_str(), fresnel * specular_direct);
	if (diffuse_indirect != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_indirect_diffuse.c_str(), Kd * diffuse_indirect);
	if (specular_indirect != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_indirect_specular.c_str(), fresnel * specular_indirect);

	sg->out.RGB = o_diffuse + o_specular;

	/////@@@ TO DO
	////	 Indirect specular V
	////	 AOVs support V
	////	 
	////	 Better Fresnel
	////	 Stage 1 - Build sample set
	////	 Stage 2 - Construct microfacet H vector
	////	 Stage 3 - Schlick approximation to Fresnel for dielectrics

	////	 Specular colour parameter (metallic?)
	////	 Normal map support
	////	
}

node_loader
{
	if (i > 0)
	return false;

	node->methods = SimpleMethods;
	node->output_type = AI_TYPE_RGB;
	node->name = "simple";
	node->node_type = AI_NODE_SHADER;
	strcpy(node->version, AI_VERSION);
	return true;
}